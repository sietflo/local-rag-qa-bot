# Authentication

## Reset password
To reset your password, open Settings → Security and click "Reset password".
A reset link is sent to your email and expires after 30 minutes.
For security, passwords expire after 90 days and must be changed.

## API authentication
Authenticate API requests with a Bearer token in the Authorization header:
`Authorization: Bearer <token>`. Create tokens in Settings → API keys.
Tokens can be revoked at any time and are never shown again after creation.

## Two-factor authentication
Two-factor authentication (2FA) can be enabled in Settings → Security.
We support authenticator apps (TOTP) and SMS codes.
