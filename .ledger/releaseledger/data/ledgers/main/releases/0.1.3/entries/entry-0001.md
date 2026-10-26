---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.3
kind: fixed
summary:
  Fixed unsupported Turbo q4f16 CPU provider requests before model acquisition
  or session creation
status: accepted
audience: null
scopes: []
source_refs:
  - git:6d1499043dd6ccb310fbb11b477aad68e73a84ed
paths:
  - README.md
  - chatterboxsynth/runtime.py
  - docs/architecture.md
  - docs/release-0.1.0.md
  - tests/test_cli.py
  - tests/test_runtime.py
issues: []
prs: []
sources:
  - git:6d1499043dd6ccb310fbb11b477aad68e73a84ed
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

Documents the provider limitation and keeps Turbo fp32 available for CPU use while adding coverage for pretrained, managed, and local loading paths.
