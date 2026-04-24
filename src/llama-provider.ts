
import * as vscode from "vscode";
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelChatMessage,
    LanguageModelChatProvider,
    ProvideLanguageModelChatResponseOptions,
    LanguageModelResponsePart,
    Progress,
} from "vscode";
import { BaseChatModelProvider, DEFAULT_CONTEXT_LENGTH, DEFAULT_MAX_OUTPUT_TOKENS } from "./base-provider";
import { convertMessages, convertTools, validateRequest } from "./utils";

/**
 * Chat model provider for Llama.cpp servers.
 * Implements the VS Code language model chat provider interface for Llama.cpp compatible APIs.
 * Handles model discovery, chat responses, and streaming from local Llama.cpp instances.
 *
 */
export class LlamaCppChatModelProvider extends BaseChatModelProvider {
    /**
     * Creates a new Llama.cpp chat model provider.
     * Initializes the provider with secret storage and user agent for API requests.
     *
     * @param secrets - VS Code secret storage for storing server URL and API key.
     * @param userAgent - User agent string to include in HTTP requests.
     */
    constructor(secrets: vscode.SecretStorage, private readonly userAgent: string) {
        super(secrets);
    }

    /**
     * Provides information about available Llama.cpp models.
     * Fetches model list from the configured server and returns model information.
     *
     * @param options - Options for the request, including error suppression.
     * @param token - Cancellation token to abort the operation.
     * @returns Promise resolving to an array of available models.
     */
    async provideLanguageModelChatInformation(
        options: { silent: boolean },
        token: CancellationToken
    ): Promise<LanguageModelChatInformation[]> {
        const serverUrl = await this.getServerUrl();
        const apiKey = await this.getApiKey(); // Optional

        try {
            const [models, props] = await Promise.all([
                this.fetchModels(serverUrl, apiKey),
                this.fetchProps(serverUrl, apiKey).catch(() => undefined),
            ]);

            // Prefer the server's actually loaded context window (n_ctx from /props)
            // over the model's training context length (n_ctx_train from /v1/models),
            // because the server may be running with a reduced context.
            const serverCtx = props?.default_generation_settings?.n_ctx;

            return models.map(model => {
                const trainCtx = model.meta?.n_ctx_train;
                // Use the smaller of (server-loaded ctx, training ctx) when both known,
                // otherwise fall back to whichever is available, then to the default.
                const contextLength =
                    typeof serverCtx === "number" && serverCtx > 0
                        ? (typeof trainCtx === "number" && trainCtx > 0
                              ? Math.min(serverCtx, trainCtx)
                              : serverCtx)
                        : typeof trainCtx === "number" && trainCtx > 0
                          ? trainCtx
                          : DEFAULT_CONTEXT_LENGTH;

                const maxOutputTokens = Math.min(
                    DEFAULT_MAX_OUTPUT_TOKENS,
                    Math.max(1, Math.floor(contextLength / 4))
                );
                const maxInputTokens = Math.max(1, contextLength - maxOutputTokens);

                const displayName = props?.model_alias && models.length === 1 ? props.model_alias : model.id;

                return {
                    id: model.id,
                    name: displayName, // Llama.cpp usually returns filename as ID
                    tooltip: `Llama.cpp model: ${model.id} (context: ${contextLength.toLocaleString()} tokens)`,
                    family: "llama-cpp",
                    version: "1.0.0",
                    maxInputTokens,
                    maxOutputTokens,
                    capabilities: {
                        toolCalling: true, // Assuming modern models support it
                        imageInput: Boolean(props?.modalities?.vision),
                    },
                };
            });
        } catch (err) {
            if (!options.silent) {
                console.error("[Llama.cpp Provider] Failed to fetch models", err);
            }
            return []; // Return empty if failed or server not running
        }
    }

    /**
     * Fetches server properties from the Llama.cpp /props endpoint.
     * Provides the actual loaded context window and other runtime settings.
     *
     * @param serverUrl - The base URL of the Llama.cpp server.
     * @param apiKey - Optional API key for authentication.
     * @returns Promise resolving to the parsed props payload or undefined.
     */
    private async fetchProps(
        serverUrl: string,
        apiKey?: string
    ): Promise<LlamaServerProps | undefined> {
        const headers: Record<string, string> = {
            "User-Agent": this.userAgent,
        };
        if (apiKey) {
            headers["Authorization"] = `Bearer ${apiKey}`;
        }

        const response = await fetch(`${serverUrl}/props`, {
            method: "GET",
            headers,
        });

        if (!response.ok) {
            // /props may be disabled on the server; treat as soft-failure.
            return undefined;
        }

        return (await response.json()) as LlamaServerProps;
    }

    /**
     * Provides a chat response from the Llama.cpp model.
     * Sends a chat completion request to the server and processes the streaming response.
     *
     * @param model - Information about the selected model.
     * @param messages - Array of chat messages for the conversation.
     * @param options - Options for the response generation.
     * @param progress - Progress callback to report response parts.
     * @param token - Cancellation token to abort the operation.
     * @returns Promise that resolves when the response is complete.
     */
    async provideLanguageModelChatResponse(
        model: LanguageModelChatInformation,
        messages: readonly LanguageModelChatMessage[],
        options: ProvideLanguageModelChatResponseOptions,
        progress: Progress<LanguageModelResponsePart>,
        token: CancellationToken
    ): Promise<void> {
        const serverUrl = await this.getServerUrl();
        const apiKey = await this.getApiKey();

        validateRequest(messages);
        const openaiMessages = convertMessages(messages);
        const toolConfig = convertTools(options);

        // Check token limits roughly
        const inputTokenCount = this.estimateMessagesTokens(messages);
        const toolTokenCount = this.estimateToolTokens(toolConfig.tools);
        const tokenLimit = Math.max(1, model.maxInputTokens);
        if (inputTokenCount + toolTokenCount > tokenLimit) {
            console.warn(
                `[Llama.cpp Provider] Message tokens (${inputTokenCount} + ${toolTokenCount}) exceed limit ${tokenLimit}`
            );
             // Proceed anyway as local models might handle it or truncate
        }

        const requestBody: Record<string, unknown> = {
            model: model.id,
            messages: openaiMessages,
            stream: true,
            max_tokens: options.modelOptions?.max_tokens ?? model.maxOutputTokens ?? 4096,
            temperature: options.modelOptions?.temperature ?? 0.7,
        };

        if (toolConfig.tools) {
            requestBody.tools = toolConfig.tools;
        }
        if (toolConfig.tool_choice) {
            requestBody.tool_choice = toolConfig.tool_choice;
        }

        const headers: Record<string, string> = {
            "Content-Type": "application/json",
            "User-Agent": this.userAgent,
        };
        if (apiKey) {
            headers["Authorization"] = `Bearer ${apiKey}`;
        }

        try {
            const response = await fetch(`${serverUrl}/v1/chat/completions`, {
                method: "POST",
                headers,
                body: JSON.stringify(requestBody),
                signal: token.isCancellationRequested ? AbortSignal.abort() : undefined,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Llama.cpp API error: ${response.status} ${response.statusText}\n${errorText}`);
            }

            if (!response.body) {
                throw new Error("No response body from Llama.cpp API");
            }

            await this.processStreamingResponse(response.body, progress, token);
        } catch (err) {
            console.error("[Llama.cpp Provider] Chat request failed", err);
            throw err;
        }
    }

    /**
     * Retrieves the configured server URL from secrets.
     * Falls back to default localhost URL if not configured.
     *
     * @returns Promise resolving to the server URL.
     */
    private async getServerUrl(): Promise<string> {
        // Default to localhost:8080 if not configured
        return (await this.secrets.get("llamacpp.serverUrl")) || "http://localhost:8080";
    }

    /**
     * Retrieves the optional API key from secrets.
     * Returns undefined if no API key is configured.
     *
     * @returns Promise resolving to the API key or undefined.
     */
    private async getApiKey(): Promise<string | undefined> {
        return await this.secrets.get("llamacpp.apiKey");
    }

    /**
     * Fetches the list of available models from the Llama.cpp server.
     * Makes a GET request to the /v1/models endpoint.
     *
     * @param serverUrl - The base URL of the Llama.cpp server.
     * @param apiKey - Optional API key for authentication.
     * @returns Promise resolving to an array of model objects.
     */
    private async fetchModels(serverUrl: string, apiKey?: string): Promise<LlamaModelEntry[]> {
        const headers: Record<string, string> = {
             "User-Agent": this.userAgent
        };
        if (apiKey) {
            headers["Authorization"] = `Bearer ${apiKey}`;
        }

        const response = await fetch(`${serverUrl}/v1/models`, {
            method: "GET",
            headers,
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
        }

        const data = (await response.json()) as { data?: LlamaModelEntry[] };
        return data.data || [];
    }
}

/**
 * Subset of the Llama.cpp /props payload we care about.
 */
interface LlamaServerProps {
    default_generation_settings?: {
        n_ctx?: number;
    };
    model_alias?: string;
    modalities?: {
        vision?: boolean;
        audio?: boolean;
    };
}

/**
 * Entry returned from Llama.cpp /v1/models.
 * The `meta` field is a Llama.cpp extension that exposes the training context length.
 */
interface LlamaModelEntry {
    id: string;
    meta?: {
        n_ctx_train?: number;
    };
}
