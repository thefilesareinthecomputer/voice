# Security policy

## Supported versions

Only the latest release gets security fixes.

## What counts

Anything that makes voice do one of these:

- reach the network outside `voice setup`;
- read or write files outside the Hugging Face cache, its state folder, its models folder and the clipboard;
- write a transcript anywhere but the clipboard, or the terminal `voice listen` prints to;
- run code from a model, or load a model family other than Kokoro and Parakeet;
- signal a process that is not its own.

A known vulnerability in a locked dependency counts too.

## Reporting a vulnerability

Report it privately on GitHub: on the repo's **Security** tab, choose **Report a vulnerability**. Include what you found, how to reproduce it, and what an attacker could do with it.

## What to expect

The maintainer replies in the private advisory, works out the fix there, and publishes the advisory with the release that fixes it.
