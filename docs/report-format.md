# Apple Intelligence Report Format Reference

Notes on the JSON schema that macOS exports from Settings > Privacy & Security > Apple Intelligence Report.

## Top-level structure

The export is a JSON object with two arrays:

```json
{
  "modelRequests": [...],
  "privateCloudComputeRequests": [...]
}
```

## modelRequests entries

Each entry describes one request to an Apple Intelligence model.

| Field | Type | What it means |
|---|---|---|
| `timestamp` | number (epoch seconds) | When the request was made |
| `identifier` | string (UUID) | Unique request ID |
| `useCase` | string | What the model was asked to do (e.g. `com.apple.SummarizationKit.mailMessage.synopsis`) |
| `prompt` | string | The user's input, often wrapped in a Swift `PromptTemplateInfo(...)` serialisation |
| `response` | string | The model's output |
| `model` | string | Which model handled the request |
| `modelVersion` | string | Model version string |
| `clientIdentifier` | string | Bundle ID of the app that made the request (e.g. `com.apple.mobilemail`) |
| `executionEnvironment` | string | `OnDevice` or `PrivateCloudCompute` |

## privateCloudComputeRequests entries

Each entry describes a Private Cloud Compute request with attestation data.

| Field | Type | What it means |
|---|---|---|
| `timestamp` | number (epoch seconds) | When the request was made |
| `requestId` | string (UUID) | Unique request ID |
| `pipelineKind` | string | Pipeline type |
| `pipelineParameters` | string or object | JSON string with adapter and model info |
| `nodes` | array | Node objects with attestation bundles |

### nodes entries

Each node carries a base64 X.509 attestation bundle proving the code running on Apple's servers is attested and verified.

| Field | Type | What it means |
|---|---|---|
| `node` | string (base64) | Node identifier |
| `nodeState` | string | `Validated` if attestation passed |
| `attestationBundle` | string (JSON) | Base64 X.509 certificate chain |

## executionEnvironment

This is the field that answers "did this leave my Mac":

- **OnDevice**: the request ran locally on your Mac. No data left the device.
- **PrivateCloudCompute**: the request ran on Apple's servers. The attestation bundle proves which code ran and that it was verified.

## Prompt format

Prompts are often not plain text. They come as Swift `PromptTemplateInfo(...)` serialisations containing:

- `templateID`: identifies the template
- `variableBindings`: key-value pairs with the actual user content
- `locale`: the user's locale

The parser extracts the readable content from `variableBindings`, preferring keys like `userPrompt`, `prompt`, `userContent`, `doc`, and `freeformStoryPromptQuery`.
