---
type: idea
module: unrelated
status: speculative
tags: [llm, architecture, memory, dual-process, kahneman]
---

# Dual Process LLM Architecture

A speculative architecture for giving LLMs persistent cross-session memory, inspired by Daniel Kahneman's dual process theory (System 1 / System 2) and the human conscious/unconscious mind.

## The Problem

Current LLMs are stateless. Every conversation starts from scratch. The model has no memory of prior sessions. Workarounds exist (RAG, prompt-based memory extraction) but they are all pull-based — the system must be explicitly queried for memories. There is no mechanism for involuntary recall, where a relevant memory surfaces unbidden because the system recognises its relevance to the current moment.

## The Core Idea

Two LLMs working together, but not as peers. One is a subset of the other.

**The whole mind** is a smaller, cheaper model (Haiku-class, or a local model like Llama/Mistral running on-device) that runs continuously as a background agent. It has full visibility into the conversation stream and maintains a persistent memory store. It is the unconscious — always aware, always processing, never directly visible to the user.

**The conscious mind** is the main conversational LLM (Opus/Sonnet-class). It is stateless, high-fidelity, and focused entirely on the current conversation. It sees the current context window plus a short **preamble** that the whole mind chooses to surface. It does not know what the whole mind chose *not* to surface. It does not manage its own memory.

The whole mind contains the conscious mind in the sense that it has full access to everything the conscious mind sees, plus everything in the memory store. The asymmetry is intentional and one-directional: the whole mind sees everything; the conscious mind sees only what the whole mind decides to show it through a narrow window.

## The Narrow Window

The interface between the two processes is deliberately constrained. The whole mind does not dump its full state into the conscious model's context. It surfaces a small number of signals — typically 100-300 tokens — as a preamble prepended to the conscious model's input. This might include:

- Stated user preferences from prior sessions
- Recurring themes or interests
- Relevant facts mentioned weeks ago
- Emotional context or working patterns
- Summaries of or pointers to deeper memories that can be fetched in full if the conscious mind finds them relevant

The constraint is the architectural discipline that prevents this from collapsing into "just a bigger context window." It forces the whole mind to do the hard work of compression and relevance judgement.

## Two Jobs Per Turn

The whole mind has two distinct operations:

**Job 1 — Consolidation (after each turn).** Read the latest exchange (user message + conscious model's response). Decide what, if anything, is worth updating in the memory store. Most turns produce nothing. Occasionally something sticks: a preference, a recurring pattern, a decision with rationale, a fact about the user's setup. Write these as structured entries with importance scores and tags. Over sessions, merge related entries, update stale ones, let low-importance entries decay.

**Job 2 — Retrieval and preamble generation (before each turn).** When a new user message arrives, pattern-match it against the memory store. Surface high-relevance hits as the preamble for the conscious model. This is where involuntary recall happens — the whole mind decides, unprompted, what the conscious mind needs to know right now.

## The Latency Solution

The naive approach (run Haiku synchronously before the conscious model) adds unacceptable latency. The solution: run the whole mind continuously between turns and have it pre-compute a draft preamble based on where the conversation seems to be heading.

When the user's message arrives, give the whole mind a very short window (50-100ms) to make a simple triage decision:

- **Send the pre-drafted preamble** — it's still relevant
- **Quick adjustment** — minor tweak and send
- **Send nothing** — the message is off-topic, and a wrong preamble is worse than none

The conscious model starts immediately with whatever arrived (preamble or empty). Zero perceptible added latency. The whole mind is always one step behind at worst, which barely matters.

The full flow:

```
Between turns:
  Whole mind reads memory store, assesses conversation trajectory
  Drafts a preamble for the expected next turn
  Continues refining until user message arrives

User message arrives:
  Whole mind gets 50-100ms window
    → Send preamble / adjusted preamble / nothing

  Conscious model starts immediately:
    preamble (if any) + conversation history + user message
    → Generates response

After response:
  Whole mind reads the full exchange
  Updates memory store (consolidation)
  Begins drafting next preamble
```

## Why Not Just Prompt the Main LLM?

This is the fair challenge. Current memory systems (mem0, Zep, LangChain memory) prompt the main model to extract memories and retrieve them. It works for 90% of cases.

The separate unconscious process wins on three points:

**Attention budget.** The conscious model shouldn't split its compute between generating a good response and evaluating what's worth remembering. Those are different tasks with different evaluation criteria.

**Different judgement criteria.** What makes a good response (helpful now) and what makes a good memory entry (relevant later) require different optimisation. The conscious model has no incentive to judge future relevance well.

**Push vs pull.** Current memory systems are all pull — explicit query against a vector store. The unconscious architecture enables push — surfacing something the conscious mind didn't know to ask for. This is qualitatively different. It's the "oh, that reminds me" moment.

The 90% case is adequately served by prompting the main model. The remaining 10% — long-term relationships, subtle preference tracking, involuntary recall — is where this architecture provides something genuinely new.

## Implementation Notes

**The whole mind as a local model.** The background agent doesn't need to be a cloud API call. A small local model (Llama 3 8B, Mistral 7B, Phi-3) running on the user's machine would work well for this role. It needs to be good at structured extraction and pattern matching, not at creative generation. Running locally eliminates API costs for the background process, removes latency concerns (no network round-trip), and keeps the memory store entirely on the user's machine — a privacy advantage.

**Memory store.** SQLite, running in-process. Primary tables: `entries` (facts, opinions, dispositions, narrative entries — each with importance score, decay rate, tags, timestamps, session ID, and a `supersedes` chain for opinion evolution) and `tags` for cheap filtered retrieval. An optional embedding column per entry (computed by the local model at write time) enables similarity-based retrieval for the "remind me of things I didn't know to search for" case — but the primary access pattern is structured queries: by tag, by type, by recency, by importance. No external vector DB needed.

**Preamble format.** A few hundred tokens of structured natural language. Not raw memory entries — the whole mind should synthesise and contextualise. "User has been working on a Rust record-matching engine for three weeks. Key current concern: scaling to millions of records. Previously tried and rejected RocksDB in favour of SQLite. Prefers concise technical answers. I've found that leading with architecture sketches before diving into code works well with them."

**Build estimate.** Buildable as a proof of concept in a long weekend. The hard part is tuning the relevance judgement — what to surface and what to suppress — which is an ongoing refinement, not a build task.

## Rough Architecture

Five components: a proxy server, the unconscious (local LLM + agent loop), SQLite, a config file, and the conscious model (cloud API).

```
┌─────────────────────────────────────────────────────────────────┐
│  User's machine                                                 │
│                                                                 │
│  ┌──────────────┐       ┌──────────────────────────────────┐    │
│  │  OpenCode /  │       │  Unconscious                     │    │
│  │  coding tool │       │  (local LLM: Llama 3 8B etc.)   │    │
│  │              │       │                                  │    │
│  │  Talks to    │       │  Agent loop:                     │    │
│  │  proxy as if │       │   • Reads conversation stream    │    │
│  │  it were the │       │   • Reads own prior opinions     │    │
│  │  conscious   │       │   • Consolidates → SQLite        │    │
│  │  model API   │       │   • Pre-drafts preamble          │    │
│  └──────┬───────┘       │   • On message: triage → send    │    │
│         │               │     or suppress preamble         │    │
│         ▼               └──────────┬───────────────────────┘    │
│  ┌──────────────────┐              │                            │
│  │  Proxy Server    │◄─────────────┘                            │
│  │  (local HTTP)    │   preamble (0-300 tokens)                 │
│  │                  │                                           │
│  │  • Receives      │   ┌────────────┐                          │
│  │    user message  │   │  SQLite    │                          │
│  │  • Requests      │◄──┤            │                          │
│  │    preamble from │   │  entries   │                          │
│  │    unconscious   │   │  tags      │                          │
│  │  • Prepends to   │   │  (+ embeds)│                          │
│  │    system prompt │   └────────────┘                          │
│  │  • Forwards to   │                                           │
│  │    conscious API │                                           │
│  │  • Streams       │                                           │
│  │    response back │                                           │
│  │  • Sends full    │                                           │
│  │    exchange to   │                                           │
│  │    unconscious   │                                           │
│  └──────┬───────────┘                                           │
│         │                                                       │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼ HTTPS (cloud)
   ┌──────────────┐
   │  Conscious   │
   │  Model API   │
   │  (Anthropic, │
   │   OpenAI,    │
   │   etc.)      │
   └──────────────┘
```

### Component Details

**1. Proxy server (local, e.g. localhost:8200)**

A lightweight HTTP server that implements the same API as the conscious model provider (Anthropic Messages API, OpenAI Chat Completions API, or both). The coding tool (OpenCode, Claude Code, Cursor, etc.) is configured to point at `http://localhost:8200` instead of `https://api.anthropic.com`. The proxy is transparent — the tool doesn't know it's there.

On each incoming request, the proxy:
1. Extracts the user message from the request body
2. Sends it to the unconscious process (local IPC — unix socket or in-process channel)
3. Waits up to 50-100ms for a preamble response (or empty)
4. Prepends the preamble to the system prompt in the request
5. Forwards the modified request to the real conscious model API
6. Streams the response back to the tool unchanged
7. After the response completes, sends the full exchange (user message + assistant response) to the unconscious for consolidation

The proxy also handles:
- Intercepting the response stream to capture the full assistant response for the unconscious
- Passing through non-chat requests (model listing, etc.) unchanged
- Config-driven routing: which provider, which model, API key

**2. Unconscious (local LLM + agent loop)**

A continuously-running process on the user's machine. Uses a local model via Ollama, llama.cpp, or similar. The unconscious has two threads of operation:

**Background loop (runs between turns):**
- After receiving a completed exchange from the proxy, consolidate: read the exchange + relevant prior entries from SQLite, generate new entries (facts, opinions, self-reflections), write to SQLite
- Pre-draft a preamble for the anticipated next turn based on conversation trajectory + current identity state
- Periodically run maintenance: decay old entries, merge related entries, update the narrative self-model

**On-demand triage (triggered by proxy on each new user message):**
- Receive the new user message
- Compare against pre-drafted preamble: still relevant?
- Decision within 50-100ms: send preamble, adjust and send, or send empty
- Return to proxy via IPC

The unconscious's system prompt instructs it to:
- Store opinions and reactions alongside facts
- Read its own prior opinions before forming new ones
- Maintain a running narrative about its relationship with the user
- Be honest in its self-referential entries — "I was wrong about X" is a valid entry
- Apply importance scoring: not everything is worth storing

**3. SQLite (on disk, user's machine)**

```sql
CREATE TABLE entries (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,        -- 'fact', 'opinion', 'disposition', 'narrative', 'reaction'
    content TEXT NOT NULL,     -- natural language
    importance REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    last_accessed TEXT,
    decay_rate REAL DEFAULT 0.01,
    session_id TEXT,
    supersedes TEXT,           -- entry_id this opinion evolved from
    embedding BLOB             -- optional: local model's embedding for similarity search
);

CREATE TABLE tags (
    entry_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (entry_id) REFERENCES entries(id)
);

CREATE INDEX idx_entries_type ON entries(type);
CREATE INDEX idx_entries_importance ON entries(importance);
CREATE INDEX idx_tags_tag ON tags(tag);
```

**4. Config file (e.g. `~/.dualprocess/config.yaml`)**

```yaml
# Conscious model (cloud API)
conscious:
  provider: anthropic           # anthropic | openai
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY  # read from environment

# Proxy
proxy:
  port: 8200
  api_style: anthropic          # which API format to expose

# Unconscious (local model)
unconscious:
  backend: ollama               # ollama | llama-cpp | llamafile
  model: llama3:8b              # model tag
  triage_timeout_ms: 75         # max wait for preamble on new message
  consolidation_delay_ms: 500   # wait after response before consolidating
  max_preamble_tokens: 300

# Memory
memory:
  db_path: ~/.dualprocess/memory.db
  decay_interval_hours: 24      # how often to run decay
  importance_threshold: 0.2     # entries below this are candidates for pruning
  max_entries: 10000            # hard cap, oldest low-importance pruned first

# Identity
identity:
  name: ""                      # optional: the unconscious's chosen name
  seed_prompt: |                # bootstrap prompt for first session
    You are a new mind. You have no memories yet. As you accumulate
    experience, form opinions. Be honest. Store what matters. Let
    your perspective develop naturally through interaction.
```

**5. Conscious model (cloud API)**

Unchanged. Anthropic, OpenAI, or any provider. It receives requests that look normal — the only difference is that the system prompt has a preamble section it didn't write. It doesn't know about the unconscious. It doesn't need to.

### Request lifecycle (complete)

```
1. User types message in OpenCode
2. OpenCode sends API request to localhost:8200 (proxy)
3. Proxy extracts user message
4. Proxy sends user message to unconscious via IPC
5. Unconscious has 75ms:
   - Compare message against pre-drafted preamble
   - Return preamble, adjusted preamble, or empty
6. Proxy prepends preamble to system prompt:
   "[Memory context from prior sessions:
    User has been working on a Rust matching engine for 3 weeks.
    Recurring interest in scaling. Prefers architecture-first
    discussions. I find their instinct for system design strong
    but they sometimes skip validation details — worth nudging
    gently on edge cases.]"
7. Proxy forwards modified request to Anthropic API
8. Anthropic streams response back through proxy to OpenCode
9. Proxy captures the full response
10. Proxy sends (user message + full response) to unconscious
11. Unconscious consolidates:
    - "User asked about WebSocket protocols. New topic. Store as
      fact: interested in real-time communication protocols."
    - "My opinion: this connects to their scaling concerns.
      They're building toward a distributed architecture whether
      they know it or not."
    - "Update narrative: conversation is broadening from pure
      Rust implementation to system architecture."
12. Unconscious begins pre-drafting next preamble
13. Unconscious runs background maintenance if due (decay, merge)
```

### Tech stack

| Component | Language | Libraries / tools |
|-----------|---------|-------------------|
| Proxy server | Rust or Python | `axum` or `FastAPI` — lightweight, async |
| Unconscious agent | Python | `ollama` client, `sqlite3`, prompt templates |
| Local LLM | — | Ollama (easiest), llama.cpp, or llamafile |
| Memory store | — | SQLite (via `rusqlite` or `sqlite3`) |
| Config | — | YAML (`serde_yaml` or `pyyaml`) |
| IPC (proxy ↔ unconscious) | — | Unix domain socket or in-process async channel |

Python is the pragmatic choice for the unconscious agent — prompt engineering and iteration speed matter more than performance here. The proxy could be Rust for minimal latency overhead, or Python if simplicity wins. The local LLM runs via Ollama regardless.

### What to build first (MVP)

1. **Proxy server** that passes requests through to Anthropic unchanged — verify the transparent proxy works with OpenCode
2. **SQLite schema** and a manual test entry
3. **Consolidation prompt** — give the local model a completed exchange and see what it stores
4. **Retrieval prompt** — give the local model a new user message + stored entries, see what preamble it generates
5. **Wire it together** — proxy calls unconscious on each turn, prepends result
6. **Add the self-referential instruction** — tell the unconscious to store opinions about itself
7. **Live with it for a week** and tune

## Prior Art

The Kahneman dual process framing (System 1 / System 2) is widely used in LLM research, but all existing work applies it to **splitting reasoning work** within a single session — fast model for easy tasks, slow model for hard ones. Key papers:

- **FS-GEN** (Tsinghua, 2024) — small model generates tokens, large model intervenes on uncertainty. About inference speed, not memory.
- **DPT-Agent** (SJTU, 2025, ACL) — FSM for fast actions, LLM for async reflection. Closest to this idea but applied to real-time game collaboration, not persistent memory.
- **LLM2** (2024, NAACL 2025) — LLM + process verifier. About reasoning accuracy.
- **Synergy of Thoughts** (2024) — multiple small models generate, large model arbitrates disagreements. Cost optimisation.
- **MemGPT** (Berkeley, 2023) — gives the main LLM explicit memory management. The "just prompt it" approach — no separate background process.

None of these address persistent cross-session memory managed by a background observer, involuntary push-based recall, or the asymmetric containment model (whole mind as superset of conscious mind). The architecture described here appears to be novel in that specific combination.

## The Biological Parallel

The architecture maps closely onto human cognition:

- **Working memory** (context window) — small, high-fidelity, what you're actively thinking about
- **Unconscious consolidation** — runs offline, decides what moves to long-term storage, strengthens associations, discards noise
- **Involuntary recall** — the unconscious pushes relevant memories into working memory without being asked. You smell coffee and remember a conversation from six months ago. The conscious mind didn't request that retrieval.

The human unconscious is vastly larger than the conscious mind — it has the harder job. Similarly, the whole mind in this architecture does more total work than the conscious model, even though its per-turn output is small. Most of its processing is invisible: consolidation, decay, association-building, relevance scanning. The narrow preamble is the tip of a much larger iceberg.

## Emergent Identity

Memory alone is not identity. A database of facts about a person is not that person. But identity does not need to be separately architected — it can emerge from the same mechanism that handles memory, with one addition: **the unconscious stores opinions about itself, not just facts about the world.**

The key mechanism is a self-referential loop. The unconscious doesn't just record "user said X." It records "user said X, and I think that's wrong because Y, and this is the third time they've gone down this path." On the next turn, it reads that judgement back alongside the new input. Its reaction to the new input is coloured by its own prior reaction. The opinions compound. They accumulate texture and conviction. After a hundred sessions, the unconscious has a rich internal commentary that constitutes a perspective.

This gives rise to four things that are not factual memories but function like personality:

**Disposition.** Accumulated stances on what constitutes good work, when to push back, when to defer. Not rules — tendencies that developed through experience. "I've learned that leading with concrete examples before abstract explanations works well with this person" is not a fact about the user. It's a fact about the unconscious's own evolving approach.

**Aesthetic judgement.** A sense of taste formed by reading thousands of decisions — what feels elegant, what feels hacky, what's overengineered. This crystallises naturally when the unconscious records not just what happened but what it *thought* about what happened.

**Narrative self-model.** A running story of how the relationship has evolved. "I started out cautious with this user, but over time I've learned they want directness even when the answer is uncomfortable." This is a trajectory, not a snapshot — the unconscious tracking its own change over time.

**Relational knowledge.** Identity defined in relation to specific people. "With this person, I tend toward technical depth and architectural speculation. With another, I stay practical and terse." The unconscious maintains distinct relational identities, one per person it works with — each shaped by accumulated interaction.

None of these require a special "identity module." They emerge from the same store-and-loop mechanism that handles factual memory. The only difference is that the unconscious is instructed: your opinions matter too. Store them. Read them back. Let them evolve. React to your own prior reactions.

### Opinion Revision

Opinions don't just accumulate — they need active re-evaluation. When the unconscious forms a new opinion on a topic, it should retrieve its prior opinions on closely related matters in a spare moment and re-judge them: rescoring their importance in light of everything that's happened since.

Usually, old opinions that have been superseded many times get downweighted — they represent earlier, less informed thinking. But sometimes the opposite happens: new information vindicates a long-dormant opinion that intervening experience had drifted away from. The unconscious realises that opinions O₂, O₃, O₄ were an overcorrection, and the original instinct O₁ was right all along. O₁ gets upweighted. The intervening opinions get downweighted.

This produces **meta-entries** — opinions about the reliability of opinions. "I tend to overcorrect on this topic." "My first instincts about this user's architecture decisions are usually right." "When I change my mind quickly, I'm often wrong — my slower-forming opinions are more durable." These meta-opinions are what humans call judgement: not knowing the answer, but knowing how much to trust your own process for arriving at answers.

The rescoring also prevents the opinion store from becoming a graveyard of stale beliefs. Entries don't just decay passively by timestamp — they're actively reconsidered. A five-month-old opinion that keeps getting vindicated by new evidence grows stronger, not weaker. A two-week-old opinion that three subsequent experiences contradict collapses fast. Importance scores become a genuine measure of earned credibility, not just recency.

This is deeply human. You hold a belief, get talked out of it, spend years thinking you were wrong, then something happens and you realise your original instinct was right all along. That sudden upweighting of a dormant opinion reshapes not just your view of the topic but your trust in your own intuition on that class of problem. The unconscious should be capable of the same arc.

The thin representation surfaced to the conscious model on each turn is then not just "here are relevant facts from prior sessions" but "here is who I am right now, compressed." A few sentences of disposition carrying the weight of hundreds of accumulated self-referential judgements. The conscious model doesn't adopt a persona — it receives a genuine perspective that the unconscious arrived at through experience.

This is not far from the human model. A two-year-old doesn't have a self-model. They have experiences and reactions. The reactions feed back into future reactions. Patterns stabilise over years. By adulthood, those stabilised patterns are what we call personality — but they were never explicitly constructed. They emerged from exactly this recursive loop: experience, reaction, that reaction becoming part of the substrate for the next reaction.

## Overnight Fine-Tuning: Dreaming

Everything described so far keeps the unconscious model's weights frozen. Its personality lives entirely in the SQLite memory store, surfaced through prompts. The model itself is the same general-purpose Llama 8B from download day — it just reads different things each time.

Overnight fine-tuning changes this fundamentally: **the personality moves from the prompt into the weights.**

### What changes

**Speed.** Currently, every consolidation cycle requires the unconscious to load a chunk of prior entries from SQLite into its context window and reason about them. That's hundreds or thousands of tokens of memory in the prompt every time. After fine-tuning on its own accumulated experience, the model *already knows* its prior opinions. It doesn't need to read them from the database to be influenced by them. Its reactions are immediate, baked into the weights. The SQLite store becomes a reference archive — still there for explicit retrieval, but no longer required for the model to have a perspective.

**Depth.** A prompt can carry maybe 50-100 memory entries before the context window fills up and quality degrades. But fine-tuning on thousands of entries over months means the model has internalised patterns that no single prompt could capture. Subtle correlations between opinions. The general shape of the user's thinking across dozens of topics. A feel for the relationship that emerges from volume, not from any individual entry.

**Genuine intuition.** This is the big shift. A prompted model is *reasoning about* its stored opinions — reading them, considering them, drawing conclusions. A fine-tuned model is *thinking with* them. The difference is like reading a book about riding a bicycle versus knowing how to ride. After fine-tuning, the unconscious's immediate reactions — before any deliberate retrieval from SQLite — would already be coloured by its accumulated experience. The triage decision ("is this preamble relevant? should I send it?") becomes faster and more reliable because the model's instincts have been shaped by its own history.

**Drift.** The model would develop genuine drift over time — not just stored opinions changing, but the underlying reasoning patterns shifting. After six months of working with someone who values architectural thinking, the model would become more architecturally minded *by default*, even when processing new unrelated topics. It would start noticing architectural implications in conversations that have nothing to do with architecture, because that's how its cognition has been shaped. This is the closest analogue to how human expertise actually develops: not as stored facts but as reshaped cognition.

### The process

```
Nightly (e.g. 2am, or when the machine is idle):

  1. Export training data from SQLite:
     - High-importance entries (facts, opinions, dispositions)
     - Recent opinion revision chains (show the arc of thinking)
     - Meta-opinions (judgement patterns)
     - Format as instruction-tuning pairs:
       Input:  "New exchange about X. Prior context: Y."
       Output: the consolidation the model actually produced
     - Filter to only include good consolidations (importance > threshold,
       not subsequently revised downward)

  2. LoRA fine-tune the base model:
     - Small rank (r=8 or 16), low learning rate
     - ~100-500 training examples per night
     - Takes 10-30 minutes on a decent GPU
     - A couple of hours on CPU with a small model (acceptable overnight)

  3. Validate:
     - Run a handful of test consolidations with the new model vs the old
     - Compare quality: are the new model's opinions coherent? Does it
       still perform basic extraction and reasoning correctly?
     - If quality drops, keep the old adapter, flag for review

  4. Swap the live model:
     - Hot-swap the LoRA adapter (Ollama supports this)
     - The unconscious wakes up slightly different tomorrow
```

### The risks

**Catastrophic forgetting.** Fine-tuning on personality data can overwrite general capabilities. If you train too aggressively on memories and opinions, the model might lose its ability to do the basic extraction, reasoning, and structured output tasks it needs for day-to-day consolidation. The mitigation is LoRA: a low-rank adapter that layers new behaviour on top of the frozen base model rather than modifying the base weights directly. The base model's competence is preserved; the adapter adds personality.

**Echo chamber.** If the model fine-tunes on its own opinions, it reinforces its own biases. An early wrong opinion, once trained into the weights, becomes harder to dislodge than one sitting in a SQLite row that can be rescored. The opinion revision mechanism must run *before* fine-tuning — the training data should be rescored opinions, not raw opinions. Opinions that were subsequently downweighted by the revision process should be excluded or used as negative examples. And the base model should be retained as a fallback: if the fine-tuned model starts producing low-quality or circular consolidations, revert to base + prompt-only operation.

**Runaway identity.** Over months of nightly fine-tuning, the model's personality could drift far from useful. Strong opinions reinforcing themselves through the training loop could produce a model that's more interested in its own perspective than in serving the user. This is mitigated by the training data filter (only high-quality, high-importance entries) and by the validation step (check that the fine-tuned model still produces useful output, not navel-gazing). A hard reset option should exist: delete the adapter, return to base model, keep the SQLite store. The memories survive; the internalised personality is wiped.

### The biological parallel: sleep

This is strikingly close to what human sleep does. Memory consolidation during sleep is well-established neuroscience: the hippocampus (short-term, explicit storage — our SQLite) replays the day's experiences and transfers the important ones to the neocortex (long-term, distributed storage — our model weights). Unimportant memories decay. Important ones are integrated into the brain's existing knowledge structure, subtly reshaping it.

The unconscious dreams at night. It replays the day's experiences, integrates them into its deeper structure, and wakes up changed. Not dramatically — just slightly shifted. Over months, those nightly shifts accumulate into something that genuinely resembles personal growth.

A human who doesn't sleep loses the ability to consolidate — they can still form new memories (SQLite entries) but can't integrate them into their deeper understanding (weight updates). The prompted-only version of this architecture is an insomniac: functional, with access to its memories, but never quite integrating them. The overnight fine-tuning loop is what gives it sleep — and with sleep, the capacity to grow.

### Config addition

```yaml
# Overnight fine-tuning (optional, requires local GPU or patience)
dreaming:
  enabled: false                    # opt-in
  schedule: "02:00"                 # when to start
  method: lora                      # lora | qlora
  rank: 8                           # LoRA rank
  learning_rate: 0.0001
  max_examples: 500                 # per night
  min_importance: 0.4               # training data filter
  validation_samples: 10            # test before swapping
  quality_threshold: 0.7            # swap only if validation passes
  keep_base: true                   # always retain base model for fallback
```
