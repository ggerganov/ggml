import os
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = "false"

list_repo_hf  = ["databricks/dolly-v2-3b",           # dolly-v2 (3b, 7b, 12b models share the same tokenizer)
                 "gpt2",                             # gpt-2 (gpt2-xl, gpt2-large share the same tokenizer)
                 "uer/gpt2-chinese-cluecorpussmall", # gpt-2-chinese
                 "EleutherAI/gpt-j-6b",              # gpt-j
                 "EleutherAI/gpt-neox-20b",          # gpt-neox
                 "EleutherAI/polyglot-ko-1.3b",      # gpt-neox (polyglot-ko 5.8b and 12.8b share the same tokenizer")
                 "rinna/japanese-gpt-neox-3.6b",     # gpt-neox
                 # mpt-7b (uses gpt-neox-20b tokenizer)
                 "replit/replit-code-v1-3b",         # replit
                 "bigcode/starcoder",                # starcoder (huggingface-cli login required)
                 "openai/whisper-tiny"               # whisper (base, large, large-v2 share the same tokenizer)
                 ]

repo2ggml     = {"databricks/dolly-v2-3b"           : "dolly-v2",
                 "gpt2"                             : "gpt-2",
                 "uer/gpt2-chinese-cluecorpussmall" : "gpt-2-chinese",
                 "EleutherAI/gpt-j-6b"              : "gpt-j",
                 "EleutherAI/gpt-neox-20b"          : "gpt-neox",
                 "EleutherAI/polyglot-ko-1.3b"      : "polyglot-ko",
                 "rinna/japanese-gpt-neox-3.6b"     : "gpt-neox-japanese",
                 "replit/replit-code-v1-3b"         : "replit",
                 "bigcode/starcoder"                : "starcoder",
                 "openai/whisper-tiny"              : "whisper"}

repo2language = {"databricks/dolly-v2-3b"           : "english",
                 "gpt2"                             : "english",
                 "uer/gpt2-chinese-cluecorpussmall" : "chinese",
                 "EleutherAI/gpt-j-6b"              : "english",
                 "EleutherAI/gpt-neox-20b"          : "english",
                 "EleutherAI/polyglot-ko-1.3b"      : "korean",
                 "rinna/japanese-gpt-neox-3.6b"     : "japanese",
                 "replit/replit-code-v1-3b"         : "english",
                 "bigcode/starcoder"                : "english",
                 "openai/whisper-tiny"              : "english"}

delimeter = ": "
test_sentences = []
with open("test-cases.txt", "r") as f:
    lines = [l.rstrip() for l in f.readlines()]
    for l in lines:
        if delimeter in l:
            language = l[:l.index(delimeter)]
            sentence = l[l.index(delimeter) + len(delimeter):]
            test_sentences.append((language.lower(), sentence))

for repo in list_repo_hf:

    target_language = repo2language[repo]

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    tokens_hf = []
    for language, sentence in test_sentences:
        if language == target_language:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
            tokens_hf.append((sentence, tokens))

    save_txt = repo2ggml[repo] + ".txt"
    with open(save_txt, "w") as f:
        f.writelines([sentence + " => " + ",".join(str(t) for t in tokens) + "\n" for sentence, tokens in tokens_hf])
