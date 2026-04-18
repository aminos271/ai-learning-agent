# Router Eval Summary

- Total: 20
- Correct: 19
- Accuracy: 95.00%
- Rewrite Changed: 5/20

## Per Class
- chat: 5/5 (100.00%)
- note_recall: 4/5 (80.00%)
- note_store: 5/5 (100.00%)
- rag: 5/5 (100.00%)

## Confusion Pairs
- note_recall -> note_store: 1

## Wrong Cases
- router-14: expected `note_recall`, got `note_store` | question: 我刚刚记了什么？ | rewritten: 帮我记一下，RAG 更像查资料，不是永久记忆。
