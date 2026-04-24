# Llama.cpp Provider for GitHub Copilot Chat

A minimal VS Code extension that connects GitHub Copilot Chat to a local Llama.cpp server.

## Requirements

- VS Code 1.104.0 or newer
- A running Llama.cpp server with OpenAI-compatible `/v1` endpoints

## Install

```sh
npm install
npm run compile
```

## Use

1. Install the extension from VSIX or load it in VS Code.
2. Run the `Llama.cpp: Manage` command.
3. Enter your Llama.cpp server URL.
4. Choose the Llama.cpp provider in Copilot Chat.

## Notes

- This repository is a fork intended for local Llama.cpp integration.

## License

See `LICENSE`.
