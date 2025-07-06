# Gptcmd-anthropic
Gptcmd-anthropic adds support for [Anthropic](https://anthropic.com)'s Claude models to [Gptcmd](https://github.com/codeofdusk/gptcmd).

[Python](https://python.org) 3.8.6 or later, Gptcmd 2.2.0 or later, and an [Anthropic API key](https://console.anthropic.com/account/keys) are required to use this package. Gptcmd-anthropic is available on PyPI, and can, for instance, be installed with `pip install gptcmd-anthropic` at a command line shell.

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
To save costs, Gptcmd-anthropic dynamically inserts [cache breakpoints](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) on the system message (if present), the final user message, and the largest messages of a conversation based on content length and number of attachments.

You may override this dynamic strategy on a per-message basis by setting the `anthropic_cache_breakpoint` metadata field:

* If set to `True`, the message will always be cached.
* If set to `False`, the message will never be cached.
* If a span of consecutive messages of the same role contains conflicting breakpoint metadata (one message set to always cache, the next set to never cache), the entire span will be cached.

For instance:

```
(claude-opus-4-20250514) user Cache me!
'Cache me!' added as user
(claude-opus-4-20250514) meta anthropic_cache_breakpoint True
anthropic_cache_breakpoint set to True on 'Cache me!'
```

## Extended thinking
You may enable extended thinking with a command like `set thinking {"type": "enabled", "budget_tokens": 1024}`. When extended thinking mode is enabled, a summary of the thinking process can be found at the `anthropic_thinking_text` metadata key on the generated assistant message (consult the Gptcmd readme for details).
