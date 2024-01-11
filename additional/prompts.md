Prompt to get the English-Chinese translations from LLMs:
```
Translate the following English sentence into Chinese: "$S_s". Only print the translation.
```

Prompt to get the Chinese-English translations from LLMs:
```
Translate the following Chinese sentence into English: "$S_s". Only print the translation.
```

Prompt to fix the buggy English-Chinese translations for LLMs:
```
Translate the following two English sentences into Chinese: "$S_s" and "$S_f".
Your original translations are : "$T_s" and "$T_f", in which the words "$FINE_GRAINED_VIOLATIONS" can be wrong.
PLease re-generate the two translations.
```

Prompt to fix the buggy Chinese-English translations for LLMs:
```
Translate the following two Chinese sentences into English: "$S_s" and "$S_f".
Your original translations are : "$T_s" and "$T_f", in which the words "$FINE_GRAINED_VIOLATIONS" can be wrong.
PLease re-generate the two translations.
```