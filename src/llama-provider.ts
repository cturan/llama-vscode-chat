
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

export class LlamaCppChatModelProvider extends BaseChatModelProvider {
    constructor(secrets: vscode.SecretStorage, private readonly userAgent: string) {
        super(secrets);
    }

    async provideLanguageModelChatInformation(
        options: { silent: boolean },
        token: CancellationToken
    ): Promise<LanguageModelChatInformation[]> {
        const serverUrl = await this.getServerUrl();
        const apiKey = await this.getApiKey(); // Optional

        try {
            const models = await this.fetchModels(serverUrl, apiKey);
            return models.map(model => ({
                id: model.id,
                name: model.id, // Llama.cpp usually returns filename as ID
                tooltip: `Llama.cpp model: ${model.id}`,
                family: "llama-cpp",
                version: "1.0.0",
                maxInputTokens: DEFAULT_CONTEXT_LENGTH - DEFAULT_MAX_OUTPUT_TOKENS, // Rough estimate or configurable
                maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
                capabilities: {
                    toolCalling: true, // Assuming modern models support it
                    imageInput: false, // Could be true for vision models, but safe default is false
                },
            }));
        } catch (err) {
            if (!options.silent) {
                console.error("[Llama.cpp Provider] Failed to fetch models", err);
            }
            return []; // Return empty if failed or server not running
        }
    }

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
            max_tokens: options.modelOptions?.max_tokens || 4096,
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

    private async getServerUrl(): Promise<string> {
        // Default to localhost:8080 if not configured
        return (await this.secrets.get("llamacpp.serverUrl")) || "http://localhost:8080";
    }

    private async getApiKey(): Promise<string | undefined> {
        return await this.secrets.get("llamacpp.apiKey");
    }

    private async fetchModels(serverUrl: string, apiKey?: string): Promise<{ id: string }[]> {
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

        const data = (await response.json()) as { data: { id: string }[] };
        return data.data || [];
    }
}
