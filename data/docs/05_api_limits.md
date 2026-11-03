# API Limits and Behavior

## Rate limits

The API allows 60 requests per minute on Free, 600 requests per minute on Pro, and 3000 requests per minute on Business.

When the limit is exceeded the API returns HTTP 429 with a Retry-After header that says how many seconds to wait. Clients should back off and retry after that time.

## Tokens

API tokens expire after 90 days. Create a new token under Settings, then Developer, before the old one expires.

## Uploads

A single upload request can carry at most 5 GB. Larger files must use multipart upload with chunks of 64 MB.

## Webhooks

Failed webhook deliveries are retried 5 times with exponential backoff. After the fifth failure the webhook is disabled and the account owner receives an email.
