# Gptcmd-anthropic
Gptcmd-anthropic adds support for [Anthropic](https://anthropic.com)'s Claude models to [Gptcmd](https://github.com/codeofdusk/gptcmd).

[Python](https://python.org) 3.8.6 or later, Gptcmd 2.0.0 or later, and an [Anthropic API key](https://console.anthropic.com/account/keys) are required to use this package. Gptcmd-anthropic is available on PyPI, and can, for instance, be installed with `pip install gptcmd-anthropic` at a command line shell.

## Configuration
To use Gptcmd-anthropic, you'll need to add a new account to your Gptcmd configuration or modify your default account. If no `api_key` is specified in your configuration, Gptcmd-anthropic uses the API key in the `ANTHROPIC_API_KEY` environment variable. An example configuration follows:

``` toml
[accounts.claude]
provider = "anthropic"
api_key = "sk-ant-xxxxx"  # Replace with your API key
# Though not required, specifying a model in your configuration, similar to
# openai and azure accounts, will use that model by default
model = "claude-3-5-sonnet-latest"
# Any additional options are passed directly to the Python Anthropic client's
# constructor for this account.
```

## Usage
If you've configured multiple accounts, the `account` command in Gptcmd can be used to switch between them:

```
(gpt-4o) account claude
Switched to account 'claude'
(claude-3-5-sonnet-latest) account default
Switched to account 'default'
(gpt-4o)
```

Consult Gptcmd's readme for additional usage instructions.

## Prompt caching
To save costs, Gptcmd-anthropic dynamically inserts [cache breakpoints](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) on the system message (if present), the final user message, and the largest messages of a conversation based on content length and number of attachments. This feature is currently not user configurable.
