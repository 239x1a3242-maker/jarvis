from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover
    DDGS = None

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover
    InferenceClient = None


@dataclass
class Settings:
    hf_token: str = ""
    planning_model: str = "LiquidAI/LFM2.5-1.2B-Instruct"
    rag_model: str = "LiquidAI/LFM2-1.2B-RAG"
    thinking_model: str = "LiquidAI/LFM2.5-1.2B-Thinking"
    user_agent: str = "autobot-architecture/1.0"
    timeout_s: float = 45.0
    memory_file: str = "autobot_memory.json"
    verbose: bool = True

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            planning_model=os.getenv("PLANNING_MODEL", "LiquidAI/LFM2.5-1.2B-Instruct"),
            rag_model=os.getenv("RAG_MODEL", "LiquidAI/LFM2-1.2B-RAG"),
            thinking_model=os.getenv("THINKING_MODEL", "LiquidAI/LFM2.5-1.2B-Thinking"),
            user_agent=os.getenv("AUTOBOT_USER_AGENT", "autobot-architecture/1.0"),
            timeout_s=float(os.getenv("REQUEST_TIMEOUT", "45")),
            memory_file=os.getenv("AUTOBOT_MEMORY_FILE", "autobot_memory.json"),
            verbose=_to_bool(os.getenv("AUTOBOT_VERBOSE", "1")),
        )


@dataclass
class Message:
    timestamp: float
    sender: str
    recipient: str
    content: str


@dataclass
class AgentContext:
    query: str
    max_steps: int = 4
    route: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "general"
    complexity: str = "medium"
    depth: str = "medium"
    execution_profile: Dict[str, Any] = field(default_factory=dict)
    intent_tags: List[str] = field(default_factory=list)
    detected_capabilities: List[str] = field(default_factory=list)
    intent_analysis: Dict[str, Any] = field(default_factory=dict)
    task_probabilities: Dict[str, float] = field(default_factory=dict)
    primary_task: str = "general_qa"
    secondary_tasks: List[str] = field(default_factory=list)
    extracted_entities: List[Dict[str, str]] = field(default_factory=list)
    urgency: str = "medium"
    ambiguity_score: float = 0.0
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    plan: str = ""
    plan_steps: List[str] = field(default_factory=list)
    candidate_plans: List[Dict[str, Any]] = field(default_factory=list)
    assignments: Dict[str, str] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    dry_run_passed: bool = True
    pev_verified: bool = False
    final_answer: str = ""
    memory_recall: List[Dict[str, Any]] = field(default_factory=list)
    semantic_facts: Dict[str, str] = field(default_factory=dict)
    graph_memory: Dict[str, Any] = field(default_factory=lambda: {"nodes": [], "edges": []})
    ensemble_views: Dict[str, str] = field(default_factory=dict)
    blackboard: Dict[str, Any] = field(
        default_factory=lambda: {"artifacts": {}, "status": {}, "messages": []}
    )
    executed_stages: List[str] = field(default_factory=list)
    skipped_stages: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)

    def post(self, sender: str, recipient: str, content: str) -> None:
        msg = Message(time.time(), sender, recipient, content)
        self.messages.append(msg)
        self.blackboard["messages"].append(
            {"ts": round(msg.timestamp, 3), "from": sender, "to": recipient, "content": content}
        )
        self.trace.append(f"{sender} -> {recipient}: {content[:180]}")


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _log(level: str, component: str, message: str, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] [{component}] {message}", flush=True)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _extract_steps(plan: str) -> List[str]:
    lines = [line.strip() for line in (plan or "").splitlines() if line.strip()]
    steps: List[str] = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\).\s-]+", "", line).strip()
        if cleaned:
            steps.append(cleaned)
    if not steps and plan:
        steps = [part.strip() for part in plan.split(".") if part.strip()]
    return steps[:12]


def _first_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    for left, right in (("{", "}"), ("[", "]")):
        s = text.find(left)
        e = text.rfind(right)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                continue
    return None


def _dedupe_sources(sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for source in sources:
        url = source.get("url", "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": source.get("title", "Untitled"), "url": url})
    return out


def _extract_entities(text: str) -> List[str]:
    raw = re.findall(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b", text or "")
    banned = {"the", "and", "for", "with", "that"}
    return [item for item in raw if item.lower() not in banned][:120]


def _extract_key_facts(text: str, max_facts: int = 20) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    for sentence in re.split(r"[.!?]\s+", text or ""):
        if ":" not in sentence:
            continue
        key, value = sentence.split(":", 1)
        key = re.sub(r"[^a-zA-Z0-9 _-]", "", key).strip().lower()
        value = value.strip()
        if key and value:
            facts[key[:60]] = value[:240]
        if len(facts) >= max_facts:
            break
    return facts


CAPABILITY_CATALOG: Dict[str, Dict[str, List[str]]] = {
    "Goal Decomposition": {
        "tags": ["workflow_automation", "analysis"],
        "keywords": ["decompose", "break down", "step by step", "workflow", "pipeline"],
    },
    "Self-Prompting": {
        "tags": ["general_qa", "code_generation"],
        "keywords": ["prompt", "refine prompt", "self prompt", "improve question"],
    },
    "API Chaining": {
        "tags": ["tool_use", "data_gathering", "workflow_automation"],
        "keywords": ["api", "webhook", "integrate", "endpoint", "chain api"],
    },
    "Perception": {
        "tags": ["general_qa"],
        "keywords": ["understand", "interpret", "summarize", "read"],
    },
    "Reasoning & Planning": {
        "tags": ["analysis"],
        "keywords": ["analyze", "reason", "plan", "strategy", "architecture"],
    },
    "Tool Use": {
        "tags": ["general_qa", "data_gathering", "tool_use"],
        "keywords": ["search", "scrape", "tool", "lookup", "fetch"],
    },
    "Memory": {
        "tags": ["general_qa", "learning_adaptation"],
        "keywords": ["remember", "previous", "history", "context", "memory"],
    },
    "Execution": {
        "tags": ["general_qa", "workflow_automation"],
        "keywords": ["execute", "run", "perform", "complete"],
    },
    "Email Management": {
        "tags": ["data_gathering", "analysis", "workflow_automation"],
        "keywords": ["email", "inbox", "mailbox", "gmail", "outlook"],
    },
    "Calendar Scheduling": {
        "tags": ["data_gathering", "analysis", "workflow_automation"],
        "keywords": ["calendar", "schedule", "meeting", "appointment", "slot"],
    },
    "Data Entry & Migration": {
        "tags": ["data_gathering", "workflow_automation"],
        "keywords": ["data entry", "migration", "migrate", "transfer data", "etl"],
    },
    "Automated Debugging": {
        "tags": ["debugging", "code_generation"],
        "keywords": ["debug", "bug", "fix error", "traceback", "stack trace"],
    },
    "Code Refactoring": {
        "tags": ["code_generation", "debugging"],
        "keywords": ["refactor", "clean code", "improve code", "restructure code"],
    },
    "IT Helpdesk": {
        "tags": ["general_qa", "debugging"],
        "keywords": ["helpdesk", "support ticket", "troubleshoot", "issue"],
    },
    "Log Analysis": {
        "tags": ["analysis", "debugging", "data_gathering"],
        "keywords": ["log analysis", "logs", "error log", "monitor logs"],
    },
    "Dependency Updates": {
        "tags": ["code_generation", "verification"],
        "keywords": ["dependency", "update package", "upgrade library", "version bump"],
    },
    "Security Patching": {
        "tags": ["debugging", "decision_making", "verification"],
        "keywords": ["security patch", "vulnerability", "cve", "patch"],
    },
    "Creating Files": {
        "tags": ["code_generation", "workflow_automation"],
        "keywords": ["create file", "new file", "generate file"],
    },
    "Modifying Files": {
        "tags": ["code_generation", "debugging"],
        "keywords": ["modify file", "edit file", "update file", "change file"],
    },
    "Executing Code Files": {
        "tags": ["tool_use", "debugging", "workflow_automation"],
        "keywords": ["run script", "execute file", "run code"],
    },
    "Error Handling": {
        "tags": ["debugging", "learning_adaptation"],
        "keywords": ["error handling", "exception", "try except", "recover"],
    },
    "Code Compilation": {
        "tags": ["code_generation", "debugging", "verification"],
        "keywords": ["compile", "build", "compiler"],
    },
    "Syntax Checking": {
        "tags": ["debugging", "code_generation", "verification"],
        "keywords": ["syntax", "lint", "static check", "parse error"],
    },
    "Exception Logging": {
        "tags": ["debugging", "data_gathering"],
        "keywords": ["exception logging", "log exception", "capture error"],
    },
    "Automated Testing": {
        "tags": ["verification", "debugging"],
        "keywords": ["test", "unit test", "integration test", "pytest", "qa check"],
    },
    "Environment Configuration": {
        "tags": ["workflow_automation", "tool_use"],
        "keywords": ["environment", "config", "setup", ".env", "deployment config"],
    },
    "Deep Web Research": {
        "tags": ["data_gathering", "analysis"],
        "keywords": ["deep research", "web research", "sources", "citations"],
    },
    "Market Monitoring": {
        "tags": ["data_gathering", "analysis", "prediction_forecasting"],
        "keywords": ["market", "monitor", "trend", "price tracking"],
    },
    "Document Comparison": {
        "tags": ["analysis", "verification"],
        "keywords": ["compare document", "difference", "diff", "contrast"],
    },
    "Sentiment Tracking": {
        "tags": ["data_gathering", "analysis", "prediction_forecasting"],
        "keywords": ["sentiment", "opinion", "tone analysis", "social sentiment"],
    },
    "Academic Paper Synthesis": {
        "tags": ["analysis", "general_qa", "data_gathering"],
        "keywords": ["paper", "research article", "literature review", "synthesize papers"],
    },
    "Competitive Intelligence": {
        "tags": ["data_gathering", "analysis"],
        "keywords": ["competitor", "competitive intelligence", "benchmark competitor"],
    },
    "Product Recommendation": {
        "tags": ["analysis", "data_gathering", "decision_making"],
        "keywords": ["recommend product", "best product", "buying advice"],
    },
    "Churn Prediction": {
        "tags": ["prediction_forecasting", "analysis"],
        "keywords": ["churn", "retention risk", "customer loss prediction"],
    },
    "Travel Planning": {
        "tags": ["data_gathering", "analysis", "workflow_automation"],
        "keywords": ["travel", "itinerary", "trip plan", "flight", "hotel"],
    },
    "Smart Home Management": {
        "tags": ["general_qa", "workflow_automation", "tool_use"],
        "keywords": ["smart home", "iot", "home automation", "device control"],
    },
    "Financial Management": {
        "tags": ["analysis", "decision_making", "prediction_forecasting"],
        "keywords": ["finance", "cash flow", "portfolio", "money management"],
    },
    "Budget Optimization": {
        "tags": ["analysis", "decision_making"],
        "keywords": ["budget", "optimize spend", "cost reduction", "allocation"],
    },
    "Learning Tutor": {
        "tags": ["general_qa", "learning_adaptation"],
        "keywords": ["teach me", "tutor", "lesson", "learning plan"],
    },
    "Social Media Management": {
        "tags": ["data_gathering", "general_qa", "workflow_automation"],
        "keywords": ["social media", "post schedule", "content calendar", "engagement"],
    },
    "Task Handoff": {
        "tags": ["collaboration", "workflow_automation"],
        "keywords": ["handoff", "delegate", "assign task", "transfer ownership"],
    },
    "Consensus Building": {
        "tags": ["collaboration", "negotiation", "decision_making"],
        "keywords": ["consensus", "align stakeholders", "agreement", "resolve conflict"],
    },
    "Resource Allocation": {
        "tags": ["decision_making", "analysis", "workflow_automation"],
        "keywords": ["resource allocation", "allocate", "prioritize resources", "capacity planning"],
    },
    "Model Self-Correction": {
        "tags": ["learning_adaptation", "debugging", "verification"],
        "keywords": ["self-correct", "self correction", "improve output", "revise answer"],
    },
}

TAG_PRIORITY: List[str] = [
    "debugging",
    "code_generation",
    "verification",
    "workflow_automation",
    "data_gathering",
    "analysis",
    "decision_making",
    "prediction_forecasting",
    "collaboration",
    "negotiation",
    "learning_adaptation",
    "tool_use",
    "general_qa",
]


TOKEN_NORMALIZATION: Dict[str, str] = {
    "debug": "debugging",
    "bug": "debugging",
    "error": "debugging",
    "errors": "debugging",
    "fix": "debugging",
    "failing": "debugging",
    "failed": "debugging",
    "script": "code",
    "scripts": "code",
    "coding": "code",
    "compile": "verification",
    "compilation": "verification",
    "lint": "verification",
    "test": "verification",
    "tests": "verification",
    "testing": "verification",
    "schedule": "calendar",
    "meeting": "calendar",
    "mail": "email",
    "emails": "email",
    "automate": "automation",
    "automation": "workflow_automation",
    "forecast": "prediction_forecasting",
    "predict": "prediction_forecasting",
}


def _normalize_token(token: str) -> str:
    t = token.strip().lower()
    if t in TOKEN_NORMALIZATION:
        t = TOKEN_NORMALIZATION[t]
    for suffix in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return TOKEN_NORMALIZATION.get(t, t)


def _normalized_tokens(text: str) -> List[str]:
    return [_normalize_token(token) for token in _tokenize(text) if len(token) > 1]


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    positive_scores = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(positive_scores.values())
    if total <= 0:
        return {key: 0.0 for key in scores}
    return {key: positive_scores[key] / total for key in scores}


def _semantic_overlap_score(query_tokens: List[str], candidate_tokens: List[str]) -> float:
    q_set = set(query_tokens)
    c_set = set(candidate_tokens)
    if not q_set or not c_set:
        return 0.0
    intersection = len(q_set.intersection(c_set))
    union = len(q_set.union(c_set))
    jaccard = intersection / max(1, union)
    containment = intersection / max(1, min(len(q_set), len(c_set)))
    return 0.55 * jaccard + 0.45 * containment


def _detect_capabilities(query: str) -> Tuple[List[str], List[str], Dict[str, float]]:
    query_tokens = _normalized_tokens(query)
    capability_scores: Dict[str, float] = {}
    tag_scores: Dict[str, float] = {}

    for capability, meta in CAPABILITY_CATALOG.items():
        capability_text = " ".join(
            [capability] + meta.get("keywords", []) + meta.get("tags", [])
        )
        capability_tokens = _normalized_tokens(capability_text)
        score = _semantic_overlap_score(query_tokens, capability_tokens)

        for phrase in meta.get("keywords", []):
            phrase_tokens = _normalized_tokens(phrase)
            phrase_score = _semantic_overlap_score(query_tokens, phrase_tokens)
            if phrase_score > 0.45:
                score += 0.25 * phrase_score

        if score >= 0.10:
            capability_scores[capability] = score
            for tag in meta.get("tags", []):
                tag_scores[tag] = tag_scores.get(tag, 0.0) + score

    if not capability_scores:
        tag_scores["general_qa"] = tag_scores.get("general_qa", 0.0) + 0.25

    detected = sorted(capability_scores.keys(), key=lambda item: capability_scores[item], reverse=True)
    ordered_tags = sorted(
        tag_scores.keys(),
        key=lambda tag: (-tag_scores[tag], TAG_PRIORITY.index(tag) if tag in TAG_PRIORITY else 999),
    )
    return detected[:12], ordered_tags, tag_scores


def _fallback_intent_inference(query: str, memory_recall: List[Dict[str, Any]]) -> Dict[str, Any]:
    detected_capabilities, ordered_tags, tag_scores = _detect_capabilities(query)
    probability_input: Dict[str, float] = {tag: tag_scores.get(tag, 0.0) for tag in TAG_PRIORITY}

    if not any(value > 0 for value in probability_input.values()):
        probability_input["general_qa"] = 0.3
    if memory_recall:
        probability_input["analysis"] = probability_input.get("analysis", 0.0) + 0.2
        probability_input["learning_adaptation"] = probability_input.get("learning_adaptation", 0.0) + 0.1

    probabilities = _softmax(probability_input)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0] if ranked else "general_qa"
    secondary = [tag for tag, score in ranked[1:4] if score >= 0.14]
    top = ranked[0][1] if ranked else 0.0
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    relative_competition = second / max(top, 0.001)
    ambiguity = max(
        0.0,
        min(1.0, (1.0 - top) * 0.65 + relative_competition * 0.35),
    )

    generic_patterns = [
        "fix this",
        "do this",
        "help me",
        "can you do it",
        "solve this",
    ]
    requires_clarification = ambiguity >= 0.74 or any(pat in query.lower() for pat in generic_patterns)

    missing_info: List[str] = []
    clarification_questions: List[str] = []
    if requires_clarification:
        missing_info = [
            "Exact objective and desired output format",
            "Environment/input data constraints",
            "Success criteria for completion",
        ]
        clarification_questions = [
            "What exact output do you want me to produce?",
            "What input, files, or environment should I operate on?",
            "How will you decide the result is correct?",
        ]

    entities = [{"type": "entity", "value": item} for item in _extract_entities(query)[:8]]
    return {
        "primary_task": primary,
        "secondary_tasks": secondary,
        "intent_tags": ordered_tags[:8],
        "task_probabilities": probabilities,
        "detected_capabilities": detected_capabilities,
        "entities": entities,
        "urgency": "high" if any(word in query.lower() for word in ["urgent", "now", "immediately"]) else "medium",
        "ambiguity_score": round(ambiguity, 3),
        "requires_clarification": requires_clarification,
        "missing_information": missing_info,
        "clarification_questions": clarification_questions,
        "reasoning_summary": "Fallback semantic inference based on capability similarity and probability distribution.",
    }


class HFModelPool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._clients: Dict[str, Any] = {}
        self._model_map = {
            "planning": settings.planning_model,
            "rag": settings.rag_model,
            "thinking": settings.thinking_model,
        }

    def _client(self, key: str) -> Optional[Any]:
        if InferenceClient is None or not self.settings.hf_token:
            _log(
                "WARN",
                "HFModelPool",
                f"Model client unavailable for '{key}'. Using local fallback responses.",
                self.settings.verbose,
            )
            return None
        if key in self._clients:
            return self._clients[key]
        self._clients[key] = InferenceClient(
            model=self._model_map[key],
            token=self.settings.hf_token,
            timeout=self.settings.timeout_s,
        )
        return self._clients[key]

    def generate(
        self,
        model_key: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 450,
        temperature: float = 0.2,
    ) -> str:
        _log(
            "INFO",
            "HFModelPool",
            f"Generate start: model_key={model_key}, max_tokens={max_tokens}",
            self.settings.verbose,
        )
        client = self._client(model_key)
        if client is None:
            _log("WARN", "HFModelPool", f"Generate fallback for model_key={model_key}", self.settings.verbose)
            return self._mock(model_key, user_prompt)
        try:
            out = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if hasattr(out, "choices") and out.choices:
                _log("INFO", "HFModelPool", f"Generate success: model_key={model_key}", self.settings.verbose)
                return (out.choices[0].message.content or "").strip()
            _log(
                "WARN",
                "HFModelPool",
                f"Generate returned non-standard response for model_key={model_key}",
                self.settings.verbose,
            )
            return str(out).strip()
        except Exception as exc:  # pragma: no cover
            _log(
                "ERROR",
                "HFModelPool",
                f"Generate error for model_key={model_key}: {type(exc).__name__}: {exc}",
                self.settings.verbose,
            )
            return f"[Fallback due to model error: {exc}] {self._mock(model_key, user_prompt)}"

    @staticmethod
    def _mock(model_key: str, prompt: str) -> str:
        if model_key == "planning":
            return (
                "1. Understand the task and constraints.\n"
                "2. Gather evidence via tools.\n"
                "3. Validate with multi-agent checks.\n"
                "4. Synthesize and verify the final answer."
            )
        if model_key == "thinking":
            return (
                '{"action":"finish","action_input":"","rationale":"Current context is sufficient."}'
            )
        return f"Draft answer synthesized from current context:\n{prompt[:700]}"


class ToolBox:
    def __init__(self, user_agent: str, timeout_s: float, verbose: bool = True):
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.verbose = verbose

    def web_search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        _log("INFO", "ToolBox", f"web_search start: query='{query[:120]}', limit={limit}", self.verbose)
        if DDGS is None:
            _log("WARN", "ToolBox", "web_search fallback: duckduckgo_search package missing.", self.verbose)
            return [
                {
                    "title": "Search unavailable",
                    "url": "",
                    "snippet": "duckduckgo_search package is missing.",
                }
            ]
        results: List[Dict[str, str]] = []
        try:
            with DDGS() as ddgs:
                for hit in ddgs.text(query, max_results=limit):
                    results.append(
                        {
                            "title": str(hit.get("title", "")).strip(),
                            "url": str(hit.get("href", "")).strip(),
                            "snippet": str(hit.get("body", "")).strip(),
                        }
                    )
            _log("INFO", "ToolBox", f"web_search success: results={len(results)}", self.verbose)
        except Exception as exc:
            _log("ERROR", "ToolBox", f"web_search error: {type(exc).__name__}: {exc}", self.verbose)
            results.append({"title": "Search error", "url": "", "snippet": str(exc)})
        return results

    def web_scrape(self, url: str, max_chars: int = 3200) -> Dict[str, str]:
        _log(
            "INFO",
            "ToolBox",
            f"web_scrape start: url='{url[:180]}', max_chars={max_chars}",
            self.verbose,
        )
        if requests is None or BeautifulSoup is None:
            _log("WARN", "ToolBox", "web_scrape fallback: requests/beautifulsoup4 missing.", self.verbose)
            return {
                "url": url,
                "title": "Scrape unavailable",
                "text": "requests/beautifulsoup4 package is missing.",
            }
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = soup.title.get_text(strip=True) if soup.title else "Untitled"
            text = " ".join(soup.get_text(separator=" ").split())
            _log(
                "INFO",
                "ToolBox",
                f"web_scrape success: title='{title[:80]}', chars={len(text[:max_chars])}",
                self.verbose,
            )
            return {"url": url, "title": title, "text": text[:max_chars]}
        except Exception as exc:
            _log("ERROR", "ToolBox", f"web_scrape error: {type(exc).__name__}: {exc}", self.verbose)
            return {"url": url, "title": "Scrape error", "text": str(exc)}


class MemoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data = {"episodes": [], "semantic": {}, "graph": {"nodes": [], "edges": []}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            _log("INFO", "MemoryStore", f"No memory file found at '{self.path}'. Starting fresh.")
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            _log("INFO", "MemoryStore", f"Memory loaded from '{self.path}'.")
        except Exception:
            self.data = {"episodes": [], "semantic": {}, "graph": {"nodes": [], "edges": []}}
            _log("ERROR", "MemoryStore", f"Memory load failed for '{self.path}'. Resetting memory state.")

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, ensure_ascii=True, indent=2), encoding="utf-8")
            _log("INFO", "MemoryStore", f"Memory saved to '{self.path}'.")
        except Exception as exc:
            _log("ERROR", "MemoryStore", f"Memory save failed: {type(exc).__name__}: {exc}")
            raise

    def recall(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = set(_tokenize(query))
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for ep in self.data.get("episodes", []):
            text = f"{ep.get('query', '')} {ep.get('summary', '')}"
            score = len(query_tokens.intersection(set(_tokenize(text))))
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for score, ep in scored[:k] if score > 0]

    def write_episode(self, query: str, summary: str, sources: List[Dict[str, str]]) -> None:
        episodes = self.data.setdefault("episodes", [])
        episodes.append(
            {"timestamp": time.time(), "query": query, "summary": summary[:1200], "sources": sources[:5]}
        )
        self.data["episodes"] = episodes[-100:]

    def update_semantic(self, facts: Dict[str, str]) -> None:
        semantic = self.data.setdefault("semantic", {})
        semantic.update(facts)
        if len(semantic) > 400:
            keys = list(semantic.keys())
            for key in keys[:-400]:
                semantic.pop(key, None)

    def update_graph(self, nodes: List[str], edges: List[Tuple[str, str]]) -> None:
        graph = self.data.setdefault("graph", {"nodes": [], "edges": []})
        merged_nodes = set(graph.get("nodes", []))
        merged_nodes.update(nodes)
        merged_edges = {tuple(edge) for edge in graph.get("edges", [])}
        merged_edges.update(edges)
        graph["nodes"] = sorted(merged_nodes)
        graph["edges"] = [list(edge) for edge in sorted(merged_edges)]


class BaseAgent:
    def __init__(self, name: str, models: HFModelPool, tools: ToolBox):
        self.name = name
        self.models = models
        self.tools = tools

    def send(self, ctx: AgentContext, recipient: str, content: str) -> None:
        ctx.post(self.name, recipient, content)
        _log("DEBUG", self.name, f"Message to {recipient}: {content[:160]}", self.models.settings.verbose)


class IntentParsingAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        memory_context = "\n".join(
            [f"- {item.get('query', '')}: {item.get('summary', '')[:220]}" for item in ctx.memory_recall[:3]]
        )
        taxonomy = ", ".join(TAG_PRIORITY)
        prompt = (
            "Analyze the user request semantically and return strict JSON with keys:\n"
            "primary_task, secondary_tasks, task_probabilities, intent_tags, detected_capabilities,\n"
            "entities, urgency, ambiguity_score, requires_clarification, missing_information,\n"
            "clarification_questions, reasoning_summary.\n"
            "task_probabilities must map taxonomy labels to probability values from 0 to 1.\n"
            f"Allowed taxonomy labels: {taxonomy}\n\n"
            f"User query:\n{ctx.query}\n\n"
            f"Recent context (episodic memory):\n{memory_context if memory_context else 'None'}"
        )
        raw = self.models.generate(
            "thinking",
            "You are an intent parser for an agentic AI platform. Output valid JSON only.",
            prompt,
            max_tokens=620,
        )
        payload = _first_json(raw)
        fallback = _fallback_intent_inference(ctx.query, ctx.memory_recall)

        if not isinstance(payload, dict):
            payload = fallback
        else:
            merged = fallback.copy()
            merged.update(payload)
            payload = merged

        probabilities_raw = payload.get("task_probabilities", {})
        probabilities: Dict[str, float] = {}
        if isinstance(probabilities_raw, dict):
            for tag in TAG_PRIORITY:
                value = probabilities_raw.get(tag, 0.0)
                try:
                    probabilities[tag] = max(0.0, float(value))
                except Exception:
                    probabilities[tag] = 0.0
        if not any(value > 0 for value in probabilities.values()):
            probabilities = fallback.get("task_probabilities", {})
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {tag: value / total_prob for tag, value in probabilities.items()}
        else:
            probabilities = fallback.get("task_probabilities", {})

        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        primary_task = str(payload.get("primary_task", "")).strip() or (ranked[0][0] if ranked else "general_qa")
        if primary_task not in TAG_PRIORITY:
            primary_task = ranked[0][0] if ranked else "general_qa"

        secondary_raw = payload.get("secondary_tasks", [])
        secondary_tasks = [str(item).strip() for item in secondary_raw] if isinstance(secondary_raw, list) else []
        if not secondary_tasks:
            secondary_tasks = [tag for tag, score in ranked[1:4] if score >= 0.14]
        secondary_tasks = [tag for tag in secondary_tasks if tag and tag != primary_task]

        top = ranked[0][1] if ranked else 0.0
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        relative_competition = second / max(top, 0.001)
        default_ambiguity = max(
            0.0,
            min(1.0, (1.0 - top) * 0.65 + relative_competition * 0.35),
        )
        try:
            ambiguity = float(payload.get("ambiguity_score", default_ambiguity))
        except Exception:
            ambiguity = default_ambiguity
        ambiguity = max(0.0, min(1.0, ambiguity))

        requires_clarification = bool(payload.get("requires_clarification", False)) or ambiguity >= 0.74
        missing_information = payload.get("missing_information", fallback.get("missing_information", []))
        clarification_questions = payload.get(
            "clarification_questions",
            fallback.get("clarification_questions", []),
        )
        if not isinstance(missing_information, list):
            missing_information = fallback.get("missing_information", [])
        if not isinstance(clarification_questions, list):
            clarification_questions = fallback.get("clarification_questions", [])

        detected_capabilities = payload.get("detected_capabilities", fallback.get("detected_capabilities", []))
        if not isinstance(detected_capabilities, list):
            detected_capabilities = fallback.get("detected_capabilities", [])

        entities_raw = payload.get("entities", [])
        entities: List[Dict[str, str]] = []
        if isinstance(entities_raw, list):
            for item in entities_raw:
                if isinstance(item, dict):
                    entities.append(
                        {"type": str(item.get("type", "entity")), "value": str(item.get("value", "")).strip()}
                    )
                elif isinstance(item, str):
                    entities.append({"type": "entity", "value": item.strip()})
        if not entities:
            entities = fallback.get("entities", [])

        urgency = str(payload.get("urgency", "medium")).strip().lower()
        if urgency not in {"low", "medium", "high"}:
            urgency = "medium"

        intent_tags_raw = payload.get("intent_tags", [])
        intent_tags = [str(item).strip() for item in intent_tags_raw] if isinstance(intent_tags_raw, list) else []
        if not intent_tags:
            intent_tags = [tag for tag, score in ranked if score >= 0.12][:8]
        if primary_task not in intent_tags:
            intent_tags = [primary_task] + [tag for tag in intent_tags if tag != primary_task]

        ctx.intent_analysis = {
            "primary_task": primary_task,
            "secondary_tasks": secondary_tasks[:5],
            "task_probabilities": probabilities,
            "intent_tags": intent_tags[:8],
            "detected_capabilities": detected_capabilities[:12],
            "entities": entities[:12],
            "urgency": urgency,
            "ambiguity_score": round(ambiguity, 3),
            "requires_clarification": requires_clarification,
            "missing_information": [str(item) for item in missing_information][:8],
            "clarification_questions": [str(item) for item in clarification_questions][:6],
            "reasoning_summary": str(payload.get("reasoning_summary", ""))[:400],
        }
        ctx.task_probabilities = probabilities
        ctx.primary_task = primary_task
        ctx.secondary_tasks = secondary_tasks[:5]
        ctx.intent_tags = intent_tags[:8]
        ctx.detected_capabilities = detected_capabilities[:12]
        ctx.extracted_entities = entities[:12]
        ctx.urgency = urgency
        ctx.ambiguity_score = round(ambiguity, 3)
        ctx.requires_clarification = requires_clarification
        ctx.clarification_questions = [str(item) for item in clarification_questions][:6]
        ctx.blackboard["artifacts"]["intent_analysis"] = ctx.intent_analysis
        self.send(
            ctx,
            "MetaControllerAgent",
            (
                "Intent parsed semantically. "
                f"primary={primary_task}, secondary={ctx.secondary_tasks}, "
                f"ambiguity={ctx.ambiguity_score}, clarify={ctx.requires_clarification}"
            ),
        )


class MetaControllerAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        if not ctx.intent_analysis:
            fallback = _fallback_intent_inference(ctx.query, ctx.memory_recall)
            ctx.intent_analysis = fallback
            ctx.task_probabilities = fallback.get("task_probabilities", {})
            ctx.primary_task = fallback.get("primary_task", "general_qa")
            ctx.secondary_tasks = fallback.get("secondary_tasks", [])
            ctx.intent_tags = fallback.get("intent_tags", [])
            ctx.detected_capabilities = fallback.get("detected_capabilities", [])
            ctx.extracted_entities = fallback.get("entities", [])
            ctx.urgency = fallback.get("urgency", "medium")
            ctx.ambiguity_score = float(fallback.get("ambiguity_score", 0.0))
            ctx.requires_clarification = bool(fallback.get("requires_clarification", False))
            ctx.clarification_questions = fallback.get("clarification_questions", [])

        probabilities = ctx.task_probabilities or {}
        primary_task = ctx.primary_task or "general_qa"
        secondary = ctx.secondary_tasks or []
        probability = lambda tag: float(probabilities.get(tag, 0.0))

        use_web = (
            probability("data_gathering") >= 0.18
            or probability("tool_use") >= 0.18
            or probability("prediction_forecasting") >= 0.2
            or "Deep Web Research" in ctx.detected_capabilities
            or "Market Monitoring" in ctx.detected_capabilities
            or "Competitive Intelligence" in ctx.detected_capabilities
            or "Academic Paper Synthesis" in ctx.detected_capabilities
        )
        needs_tot = (
            probability("analysis") >= 0.18
            or probability("debugging") >= 0.2
            or probability("decision_making") >= 0.2
            or probability("verification") >= 0.2
            or len(secondary) >= 2
        )
        needs_ensemble = (
            probability("analysis") >= 0.18
            or probability("collaboration") >= 0.2
            or probability("negotiation") >= 0.2
            or probability("prediction_forecasting") >= 0.2
            or len(secondary) >= 2
        )
        verification_depth = (
            "high"
            if probability("verification") >= 0.2
            or probability("debugging") >= 0.22
            or use_web
            else "medium"
        )

        ctx.task_type = primary_task
        ctx.route = {
            "use_web": use_web,
            "verification_depth": verification_depth,
            "needs_tree_of_thoughts": needs_tot,
            "needs_ensemble": needs_ensemble,
            "task_type": primary_task,
            "task_category": primary_task,
            "intent_tags": ctx.intent_tags,
            "capabilities": ctx.detected_capabilities,
            "task_probabilities": probabilities,
            "primary_task": primary_task,
            "secondary_tasks": secondary,
            "requires_clarification": ctx.requires_clarification,
            "urgency": ctx.urgency,
            "ambiguity_score": ctx.ambiguity_score,
        }
        self.send(ctx, "PlanningAgent", f"Route selected: {ctx.route}")


class EpisodicSemanticAgent(BaseAgent):
    def recall(self, ctx: AgentContext, store: MemoryStore) -> None:
        ctx.memory_recall = store.recall(ctx.query, k=3)
        if ctx.memory_recall:
            self.send(ctx, "PlanningAgent", f"Recalled {len(ctx.memory_recall)} past episodes.")
        else:
            self.send(ctx, "PlanningAgent", "No relevant episode found.")

    def write(self, ctx: AgentContext, store: MemoryStore) -> None:
        facts = _extract_key_facts(ctx.final_answer)
        ctx.semantic_facts.update(facts)
        store.write_episode(ctx.query, ctx.final_answer, _dedupe_sources(ctx.sources))
        store.update_semantic(facts)
        store.update_graph(
            ctx.graph_memory.get("nodes", []),
            [tuple(edge) for edge in ctx.graph_memory.get("edges", [])],
        )
        store.save()
        self.send(ctx, "BlackboardAgent", "Episodic + semantic + graph memory updated.")


class PlanningAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        recall_text = "\n".join(
            [f"- Query: {ep.get('query', '')} | Summary: {ep.get('summary', '')[:180]}" for ep in ctx.memory_recall]
        )
        prompt = (
            f"User query:\n{ctx.query}\n\n"
            f"Route:\n{json.dumps(ctx.route, ensure_ascii=True)}\n\n"
            f"Relevant memory:\n{recall_text if recall_text else 'None'}\n\n"
            "Return a concise numbered execution plan."
        )
        ctx.plan = self.models.generate(
            "planning",
            "You are the planning agent for an advanced multi-agent system.",
            prompt,
            max_tokens=380,
        ).strip()
        ctx.plan_steps = _extract_steps(ctx.plan)
        ctx.blackboard["artifacts"]["plan"] = ctx.plan
        self.send(ctx, "TreeOfThoughtsAgent", f"Created plan with {len(ctx.plan_steps)} step(s).")


class GoalDecompositionAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        prompt = (
            f"User query:\n{ctx.query}\n\n"
            f"Task category: {ctx.task_type}\n"
            f"Intent tags: {', '.join(ctx.intent_tags) if ctx.intent_tags else 'none'}\n"
            f"Detected capabilities: {', '.join(ctx.detected_capabilities) if ctx.detected_capabilities else 'none'}\n\n"
            "Return 3 to 8 compact sub-goals as a JSON array of strings, ordered by execution."
        )
        raw = self.models.generate(
            "planning",
            "You are a goal decomposition specialist for agentic systems. Prefer strict JSON.",
            prompt,
            max_tokens=260,
        )
        payload = _first_json(raw)
        subtasks: List[str] = []
        if isinstance(payload, list):
            subtasks = [str(item).strip() for item in payload if str(item).strip()]
        elif isinstance(payload, dict):
            values = payload.get("subtasks") or payload.get("steps") or []
            subtasks = [str(item).strip() for item in values if str(item).strip()]

        if not subtasks:
            subtasks = ctx.plan_steps[:6] if ctx.plan_steps else [
                "Understand task objective and constraints",
                "Gather required evidence and context",
                "Execute reasoning and produce final answer",
            ]

        ctx.subtasks = subtasks[:8]
        ctx.blackboard["artifacts"]["subtasks"] = ctx.subtasks
        self.send(ctx, "MultiAgentCoordinator", f"Subtasks created: {ctx.subtasks}")


class TreeOfThoughtsAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        if not ctx.route.get("needs_tree_of_thoughts", True):
            return
        prompt = (
            f"Query: {ctx.query}\n"
            f"Base plan:\n{ctx.plan}\n\n"
            "Generate 3 thought branches as JSON array with keys: approach, score, risk."
        )
        raw = self.models.generate(
            "thinking",
            "You are a Tree-of-Thoughts explorer. Prefer strict JSON.",
            prompt,
            max_tokens=420,
        )
        payload = _first_json(raw)
        branches: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            candidates = payload.get("branches", [])
        else:
            candidates = []
        for item in candidates:
            if isinstance(item, dict):
                branches.append(
                    {
                        "approach": str(item.get("approach", "")).strip(),
                        "score": float(item.get("score", 0.0) or 0.0),
                        "risk": str(item.get("risk", "medium")).strip(),
                    }
                )
        if not branches:
            branches = [
                {"approach": "Evidence-first execution", "score": 0.78, "risk": "medium"},
                {"approach": "Fast answer then verify", "score": 0.62, "risk": "high"},
                {"approach": "Debate then synthesize", "score": 0.74, "risk": "medium"},
            ]
        branches.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        ctx.candidate_plans = branches
        ctx.blackboard["artifacts"]["tot_best"] = branches[0]
        self.send(ctx, "MultiAgentCoordinator", f"Best ToT approach: {branches[0]['approach']}")


class CellularAutomataAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        steps = ctx.plan_steps or ["Understand task", "Gather evidence", "Synthesize final answer"]
        n = max(3, min(9, len(steps)))
        cells = [0.2] * n
        for i, step in enumerate(steps[:n]):
            s = step.lower()
            if "search" in s or "evidence" in s:
                cells[i] = 0.6
            elif "verify" in s or "check" in s:
                cells[i] = 0.5
            else:
                cells[i] = 0.35
        for _ in range(4):
            nxt = cells[:]
            for i in range(n):
                left = cells[i - 1] if i > 0 else cells[i]
                right = cells[i + 1] if i < n - 1 else cells[i]
                nxt[i] = max(0.0, min(1.0, 0.2 * cells[i] + 0.4 * left + 0.4 * right))
            cells = nxt
        ranked = sorted([(i, score) for i, score in enumerate(cells)], key=lambda x: x[1], reverse=True)
        priorities = [steps[i] for i, _ in ranked]
        ctx.blackboard["artifacts"]["automata_priorities"] = priorities
        self.send(ctx, "MultiAgentCoordinator", "Computed plan priorities using cellular automata.")


class MultiAgentCoordinator(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        priorities = ctx.blackboard["artifacts"].get("automata_priorities", ctx.plan_steps)
        roles = ["ToolUseAgent", "ReActAgent", "PEVAgent", "EnsembleAgent", "GraphAgent"]
        assignments: Dict[str, str] = {}
        for i, role in enumerate(roles):
            assignments[role] = priorities[i % len(priorities)] if priorities else "General support"
        ctx.assignments = assignments
        ctx.blackboard["artifacts"]["assignments"] = assignments
        self.send(ctx, "AllAgents", f"Assignments shared: {assignments}")


class ToolUseAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        if not ctx.route.get("use_web", False):
            self.send(ctx, "ReActAgent", "Tool phase skipped by policy.")
            return
        query_prompt = (
            f"User query: {ctx.query}\nPlan:\n{ctx.plan}\n"
            "Return only one focused web-search query."
        )
        search_query = self.models.generate(
            "planning",
            "You produce compact search queries.",
            query_prompt,
            max_tokens=60,
        ).splitlines()[0].strip()
        if (
            not search_query
            or len(search_query) > 120
            or search_query.lower().startswith("1.")
            or "understand the task" in search_query.lower()
        ):
            search_query = ctx.query

        search_results = self.tools.web_search(search_query, limit=5)
        ctx.observations.append(
            "WEB_SEARCH:\n"
            + "\n".join(
                [f"{i + 1}. {r['title']} | {r['url']} | {r['snippet']}" for i, r in enumerate(search_results)]
            )
        )
        self.send(ctx, "BlackboardAgent", f"Search done for: {search_query}")

        scraped = 0
        for item in search_results:
            if scraped >= 2:
                break
            url = item.get("url", "")
            if not url.startswith("http"):
                continue
            page = self.tools.web_scrape(url, max_chars=3200)
            ctx.observations.append(f"SCRAPE: {page.get('title', '')}\n{page.get('text', '')}")
            ctx.sources.append({"title": page.get("title", "Untitled"), "url": page.get("url", "")})
            scraped += 1
        self.send(ctx, "ReActAgent", f"Tool-use gathered {len(ctx.sources)} source(s).")


class ReActAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        for step in range(ctx.max_steps):
            latest = "\n\n".join(ctx.observations[-2:]) if ctx.observations else "No observations yet."
            prompt = (
                "Allowed actions: search, scrape, note, finish. "
                "Return strict JSON: action, action_input, rationale.\n"
                f"Query: {ctx.query}\n"
                f"Latest observations: {latest[:1700]}"
            )
            raw = self.models.generate(
                "thinking",
                "You are a ReAct controller.",
                prompt,
                max_tokens=180,
            )
            payload = _first_json(raw)
            if isinstance(payload, dict):
                action = str(payload.get("action", "finish")).lower()
                action_input = str(payload.get("action_input", ""))
                rationale = str(payload.get("rationale", ""))
            else:
                action, action_input, rationale = "finish", "", "Unparseable action."
            self.send(ctx, "BlackboardAgent", f"ReAct step {step + 1}: {action} ({rationale[:100]})")

            if action == "finish":
                break
            if action == "note":
                if action_input.strip():
                    ctx.observations.append(f"REACT_NOTE: {action_input.strip()}")
                continue
            if action == "search":
                query = action_input.strip() or ctx.query
                results = self.tools.web_search(query, limit=3)
                ctx.observations.append(
                    "REACT_SEARCH:\n"
                    + "\n".join([f"- {r['title']} | {r['url']} | {r['snippet']}" for r in results])
                )
                continue
            if action == "scrape":
                url = action_input.strip()
                if url.startswith("http"):
                    page = self.tools.web_scrape(url, max_chars=3000)
                    ctx.observations.append(f"REACT_SCRAPE: {page.get('title', '')}\n{page.get('text', '')}")
                    ctx.sources.append({"title": page.get("title", "Untitled"), "url": page.get("url", "")})
                else:
                    ctx.observations.append("REACT_SCRAPE skipped: invalid URL.")


class BlackboardAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        summary = {
            "observations": len(ctx.observations),
            "sources": len(_dedupe_sources(ctx.sources)),
            "messages": len(ctx.messages),
            "assignments": ctx.assignments,
        }
        ctx.blackboard["status"] = summary
        ctx.blackboard["artifacts"]["latest_observation"] = ctx.observations[-1] if ctx.observations else ""
        self.send(ctx, "PEVAgent", f"Blackboard updated: {summary}")


class GraphAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        text = " ".join([ctx.query] + ctx.observations[-4:])
        entities = _extract_entities(text)
        edges: List[Tuple[str, str]] = []
        for i in range(len(entities)):
            for j in range(i + 1, min(i + 3, len(entities))):
                if entities[i] != entities[j]:
                    edges.append((entities[i], entities[j]))
        edges = edges[:80]
        ctx.graph_memory = {"nodes": sorted(set(entities)), "edges": [list(edge) for edge in edges]}
        self.send(ctx, "EnsembleAgent", f"Graph built with {len(ctx.graph_memory['nodes'])} nodes.")


class EnsembleAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        if not ctx.route.get("needs_ensemble", True):
            return
        base = f"Query: {ctx.query}\nPlan: {ctx.plan}\nEvidence count: {len(ctx.observations)}"
        views = {
            "analyst": self.models.generate(
                "thinking",
                "You are an analyst. Focus on evidence quality.",
                base,
                max_tokens=180,
            ),
            "skeptic": self.models.generate(
                "thinking",
                "You are a skeptic. Focus on gaps and risks.",
                base,
                max_tokens=180,
            ),
            "builder": self.models.generate(
                "thinking",
                "You are a builder. Focus on practical delivery.",
                base,
                max_tokens=180,
            ),
        }
        ctx.ensemble_views = {k: v.strip() for k, v in views.items()}
        ctx.blackboard["artifacts"]["ensemble"] = ctx.ensemble_views
        self.send(ctx, "MentalLoopAgent", "Ensemble debate completed.")


class MentalLoopAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        risk = 0.15
        if ctx.route.get("use_web", False) and not _dedupe_sources(ctx.sources):
            risk += 0.45
        if len(ctx.observations) < 2:
            risk += 0.2
        if ctx.warnings:
            risk += 0.1
        risk = max(0.0, min(1.0, risk))
        simulation = {
            "risk_score": round(risk, 3),
            "recommendation": "proceed" if risk < 0.55 else "gather_more_evidence",
        }
        ctx.blackboard["artifacts"]["mental_loop"] = simulation
        self.send(ctx, "DryRunAgent", f"Mental simulation output: {simulation}")


class DryRunAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        sim = ctx.blackboard["artifacts"].get("mental_loop", {})
        risk = float(sim.get("risk_score", 0.4))
        passed = risk < 0.75
        if ctx.route.get("use_web", False) and not _dedupe_sources(ctx.sources):
            passed = False
        ctx.dry_run_passed = passed
        if not passed:
            ctx.warnings.append("Dry-run failed. Evidence quality is not enough.")
        self.send(ctx, "PEVAgent", f"Dry-run status: {'pass' if passed else 'fail'}")


class PEVAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        steps = ctx.plan_steps or _extract_steps(ctx.plan)
        hay = " ".join(ctx.observations).lower()
        covered = 0
        for step in steps[:8]:
            tokens = [t for t in _tokenize(step) if len(t) > 3]
            if any(token in hay for token in tokens):
                covered += 1
        coverage = covered / max(1, min(len(steps), 8))
        ctx.pev_verified = coverage >= 0.35 and ctx.dry_run_passed
        ctx.blackboard["status"]["pev"] = {"coverage": round(coverage, 3), "verified": ctx.pev_verified}
        self.send(ctx, "SynthesisEngine", f"PEV result: {ctx.blackboard['status']['pev']}")


class RLHFAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        critique_prompt = (
            f"Draft answer:\n{ctx.final_answer}\n\n"
            "Give compact critique with factual, clarity, and actionability issues."
        )
        critique = self.models.generate(
            "thinking",
            "You are an editor/critic.",
            critique_prompt,
            max_tokens=240,
        )
        rewrite_prompt = (
            f"Improve the answer using this critique.\nCritique:\n{critique}\n\n"
            f"Draft:\n{ctx.final_answer}\n\nReturn improved final answer."
        )
        revised = self.models.generate(
            "planning",
            "You revise outputs using feedback.",
            rewrite_prompt,
            max_tokens=600,
        )
        if revised.strip():
            ctx.final_answer = revised.strip()
        self.send(ctx, "ReflexiveMetacognitiveAgent", "RLHF loop completed.")


class ReflexiveMetacognitiveAgent(BaseAgent):
    def run(self, ctx: AgentContext) -> None:
        confidence = 0.4
        if ctx.pev_verified:
            confidence += 0.25
        if ctx.dry_run_passed:
            confidence += 0.15
        confidence += min(0.2, len(_dedupe_sources(ctx.sources)) * 0.05)
        confidence = max(0.0, min(1.0, confidence))
        ctx.confidence = round(confidence, 3)
        if confidence < 0.55:
            ctx.warnings.append(
                "Metacognitive warning: confidence is moderate/low. Add targeted retrieval for higher reliability."
            )
        self.send(ctx, "FinalOutput", f"Final confidence: {ctx.confidence}")


class AdvancedAutoBotArchitecture:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = HFModelPool(settings)
        self.tools = ToolBox(settings.user_agent, settings.timeout_s, verbose=settings.verbose)
        self.memory = MemoryStore(settings.memory_file)

        self.intent_parser = IntentParsingAgent("IntentParsingAgent", self.models, self.tools)
        self.meta_controller = MetaControllerAgent("MetaControllerAgent", self.models, self.tools)
        self.episodic_semantic = EpisodicSemanticAgent("EpisodicSemanticAgent", self.models, self.tools)
        self.planning = PlanningAgent("PlanningAgent", self.models, self.tools)
        self.goal_decomposition = GoalDecompositionAgent("GoalDecompositionAgent", self.models, self.tools)
        self.tree_of_thoughts = TreeOfThoughtsAgent("TreeOfThoughtsAgent", self.models, self.tools)
        self.cellular_automata = CellularAutomataAgent("CellularAutomataAgent", self.models, self.tools)
        self.multi_agent = MultiAgentCoordinator("MultiAgentCoordinator", self.models, self.tools)
        self.tool_use = ToolUseAgent("ToolUseAgent", self.models, self.tools)
        self.react = ReActAgent("ReActAgent", self.models, self.tools)
        self.blackboard = BlackboardAgent("BlackboardAgent", self.models, self.tools)
        self.graph = GraphAgent("GraphAgent", self.models, self.tools)
        self.ensemble = EnsembleAgent("EnsembleAgent", self.models, self.tools)
        self.mental_loop = MentalLoopAgent("MentalLoopAgent", self.models, self.tools)
        self.dry_run = DryRunAgent("DryRunAgent", self.models, self.tools)
        self.pev = PEVAgent("PEVAgent", self.models, self.tools)
        self.rlhf = RLHFAgent("RLHFAgent", self.models, self.tools)
        self.reflexive = ReflexiveMetacognitiveAgent(
            "ReflexiveMetacognitiveAgent", self.models, self.tools
        )
        _log("INFO", "Orchestrator", "AdvancedAutoBotArchitecture initialized.", self.settings.verbose)

    def _select_execution_profile(self, ctx: AgentContext) -> Dict[str, Any]:
        query_tokens = _normalized_tokens(ctx.query)
        category = ctx.route.get("task_category", ctx.primary_task or ctx.task_type or "general_qa")
        probabilities = ctx.task_probabilities or {}
        if not probabilities:
            fallback = _fallback_intent_inference(ctx.query, ctx.memory_recall)
            probabilities = fallback.get("task_probabilities", {})
            ctx.task_probabilities = probabilities

        probability = lambda tag: float(probabilities.get(tag, 0.0))
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_prob = ranked[0][1] if ranked else 0.0
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        strong_intents = [tag for tag, score in ranked if score >= 0.14]
        secondary_tasks = ctx.secondary_tasks or [tag for tag in strong_intents if tag != category][:4]

        score = 0.0
        score += 0.8 if len(query_tokens) > 12 else 0.0
        score += 0.8 if len(query_tokens) > 24 else 0.0
        score += 0.7 if len(secondary_tasks) >= 2 else 0.0
        score += 0.8 if len(strong_intents) >= 3 else 0.0
        score += 2.2 * probability("debugging")
        score += 1.8 * probability("analysis")
        score += 1.6 * probability("workflow_automation")
        score += 1.5 * probability("verification")
        score += 1.4 * probability("decision_making")
        score += 1.3 * probability("prediction_forecasting")
        score += 1.2 * probability("collaboration")
        score += 1.0 * probability("data_gathering")
        score += 0.6 if ctx.route.get("use_web", False) else 0.0
        score += 0.5 if ctx.urgency == "high" else 0.0
        if category == "general_qa" and top_prob >= 0.70 and second_prob < 0.20 and len(query_tokens) <= 12:
            score -= 1.2

        if score <= 2.2:
            complexity = "simple"
            depth = "low"
        elif score <= 5.0:
            complexity = "medium"
            depth = "medium"
        else:
            complexity = "hard"
            depth = "high"

        run_clarification = bool(ctx.requires_clarification or ctx.ambiguity_score >= 0.74)

        run_tool_use = bool(
            ctx.route.get("use_web", False)
            or probability("tool_use") >= 0.16
            or probability("data_gathering") >= 0.16
            or probability("prediction_forecasting") >= 0.16
        )
        run_episodic = bool(complexity != "simple" or bool(ctx.memory_recall))
        run_goal_decomposition = bool(
            complexity != "simple" or probability("workflow_automation") >= 0.14 or len(secondary_tasks) >= 2
        )
        run_tot = bool(
            complexity == "hard"
            or probability("analysis") >= 0.18
            or probability("debugging") >= 0.20
            or probability("decision_making") >= 0.18
            or probability("prediction_forecasting") >= 0.18
        )
        run_cellular = bool(
            complexity == "hard"
            or probability("workflow_automation") >= 0.20
            or probability("decision_making") >= 0.20
        )
        run_multi_agent = bool(
            complexity != "simple"
            or probability("collaboration") >= 0.16
            or probability("negotiation") >= 0.16
            or len(secondary_tasks) >= 2
        )
        run_react = bool(
            run_tool_use
            or complexity != "simple"
            or probability("debugging") >= 0.18
            or probability("code_generation") >= 0.18
            or probability("verification") >= 0.18
        )
        run_blackboard = bool(run_react or run_tool_use or run_multi_agent)
        run_graph = bool(
            complexity == "hard"
            or probability("analysis") >= 0.18
            or probability("data_gathering") >= 0.18
            or probability("prediction_forecasting") >= 0.18
        )
        run_ensemble = bool(
            complexity == "hard"
            or probability("analysis") >= 0.20
            or probability("decision_making") >= 0.18
            or probability("collaboration") >= 0.18
            or len(secondary_tasks) >= 2
        )
        run_mental = bool(
            complexity == "hard"
            or probability("debugging") >= 0.20
            or probability("decision_making") >= 0.20
        )
        run_dry_run = bool(
            complexity == "hard"
            or probability("verification") >= 0.18
            or probability("debugging") >= 0.18
            or probability("workflow_automation") >= 0.20
        )
        run_pev = bool(complexity != "simple" or run_tool_use or probability("verification") >= 0.16)
        run_rlhf = bool(
            complexity == "hard"
            or probability("analysis") >= 0.18
            or probability("code_generation") >= 0.18
            or probability("learning_adaptation") >= 0.16
        )
        run_reflexive = True
        parallel_analysis = bool(
            complexity == "hard"
            and run_graph
            and run_ensemble
            and (probability("analysis") + probability("data_gathering") + probability("prediction_forecasting") >= 0.30)
        )
        allow_retry_evidence = bool((complexity in {"medium", "hard"}) and run_tool_use and run_pev)
        react_steps_cap = 1 if complexity == "simple" else 3 if complexity == "medium" else 6
        if probability("debugging") >= 0.22 or probability("verification") >= 0.22:
            react_steps_cap = max(react_steps_cap, 4 if complexity == "medium" else 6)

        if run_clarification:
            run_goal_decomposition = False
            run_tot = False
            run_cellular = False
            run_multi_agent = False
            run_tool_use = False
            run_react = False
            run_blackboard = False
            run_graph = False
            run_ensemble = False
            run_mental = False
            run_dry_run = False
            run_pev = False
            run_rlhf = False
            parallel_analysis = False
            allow_retry_evidence = False
            react_steps_cap = 1

        profile = {
            "task_category": category,
            "complexity": complexity,
            "depth": depth,
            "intent_tags": ctx.intent_tags,
            "detected_capabilities": ctx.detected_capabilities,
            "task_probabilities": probabilities,
            "primary_task": ctx.primary_task,
            "secondary_tasks": secondary_tasks,
            "run_clarification": run_clarification,
            "clarification_reason": "High ambiguity and low intent separation" if run_clarification else "",
            "run_episodic": run_episodic,
            "run_planning": True,
            "run_goal_decomposition": run_goal_decomposition,
            "run_tree_of_thoughts": run_tot,
            "run_cellular_automata": run_cellular,
            "run_multi_agent": run_multi_agent,
            "run_tool_use": run_tool_use,
            "run_react": run_react,
            "run_blackboard": run_blackboard,
            "run_graph": run_graph,
            "run_ensemble": run_ensemble,
            "run_mental_loop": run_mental,
            "run_dry_run": run_dry_run,
            "run_pev": run_pev,
            "run_rlhf": run_rlhf,
            "run_reflexive": run_reflexive,
            "parallel_analysis": parallel_analysis,
            "allow_retry_evidence": allow_retry_evidence,
            "react_steps_cap": react_steps_cap,
        }

        ctx.complexity = complexity
        ctx.depth = depth
        ctx.execution_profile = profile
        ctx.route.update(
            {
                "task_category": category,
                "task_type": category,
                "complexity": complexity,
                "depth": depth,
                "parallel_analysis": parallel_analysis,
                "needs_tree_of_thoughts": run_tot,
                "needs_ensemble": run_ensemble,
                "run_clarification": run_clarification,
            }
        )
        ctx.max_steps = max(1, min(ctx.max_steps, react_steps_cap))
        ctx.blackboard["artifacts"]["execution_profile"] = profile
        ctx.post(
            "Orchestrator",
            "AllAgents",
            f"Execution profile selected: complexity={complexity}, category={category}, parallel={parallel_analysis}",
        )
        _log(
            "INFO",
            "Orchestrator",
            (
                "Execution profile -> "
                f"complexity={complexity}, category={category}, "
                f"tool_use={run_tool_use}, react={run_react}, "
                f"parallel_analysis={parallel_analysis}, clarify={run_clarification}"
            ),
            self.settings.verbose,
        )
        return profile

    def _skip_stage(self, ctx: AgentContext, stage_name: str, reason: str) -> None:
        ctx.skipped_stages.append(stage_name)
        ctx.post("Orchestrator", stage_name, f"SKIPPED: {reason}")
        _log("INFO", "Orchestrator", f"SKIP stage: {stage_name} ({reason})", self.settings.verbose)

    def _run_or_skip(
        self,
        ctx: AgentContext,
        stage_name: str,
        should_run: bool,
        fn: Callable[[], None],
        skip_reason: str,
        replan_on_error: bool = True,
    ) -> bool:
        if not should_run:
            self._skip_stage(ctx, stage_name, skip_reason)
            return True
        return self._run_stage(ctx, stage_name, fn, replan_on_error=replan_on_error)

    def _run_parallel_group(
        self,
        ctx: AgentContext,
        group_name: str,
        stages: List[Tuple[str, Callable[[], None]]],
        replan_on_error: bool = True,
    ) -> Dict[str, bool]:
        _log(
            "INFO",
            "Orchestrator",
            f"START parallel group: {group_name} with stages={[name for name, _ in stages]}",
            self.settings.verbose,
        )
        results: Dict[str, bool] = {}
        if not stages:
            return results

        with ThreadPoolExecutor(max_workers=len(stages)) as executor:
            future_map = {
                executor.submit(self._run_stage, ctx, stage_name, fn, False): stage_name
                for stage_name, fn in stages
            }
            for future in as_completed(future_map):
                stage_name = future_map[future]
                try:
                    results[stage_name] = bool(future.result())
                except Exception as exc:
                    self._record_error(ctx, f"{stage_name}.parallel", exc)
                    results[stage_name] = False

        failed = [name for name, ok in results.items() if not ok]
        if failed and replan_on_error:
            self._replan_after_error(
                ctx,
                f"parallel_group:{group_name}",
                f"One or more parallel stages failed: {', '.join(failed)}",
            )
        _log(
            "INFO",
            "Orchestrator",
            f"END parallel group: {group_name}, results={results}",
            self.settings.verbose,
        )
        return results

    def _record_error(self, ctx: AgentContext, stage_name: str, exc: Exception) -> str:
        error_text = f"{type(exc).__name__}: {exc}"
        error_item = {"stage": stage_name, "error": error_text, "time": str(time.time())}
        ctx.errors.append(error_item)
        ctx.warnings.append(f"{stage_name} failed: {error_text}")
        ctx.blackboard.setdefault("errors", []).append(error_item)
        ctx.post("Orchestrator", "PlanningAgent", f"ERROR in {stage_name}: {error_text}")
        _log("ERROR", "Orchestrator", f"{stage_name} failed: {error_text}", self.settings.verbose)
        return error_text

    def _replan_after_error(self, ctx: AgentContext, stage_name: str, error_text: str) -> None:
        _log(
            "WARN",
            "Replan",
            f"Triggering replan after {stage_name} error.",
            self.settings.verbose,
        )
        try:
            replan_prompt = (
                f"User query:\n{ctx.query}\n\n"
                f"Current plan:\n{ctx.plan if ctx.plan else 'None'}\n\n"
                f"Failed stage: {stage_name}\n"
                f"Error details: {error_text}\n\n"
                f"Recent observations:\n{chr(10).join(ctx.observations[-3:]) if ctx.observations else 'None'}\n\n"
                "Create a corrected, concise numbered replan so agents can continue execution safely."
            )
            new_plan = self.models.generate(
                "planning",
                "You are a recovery planner in an autonomous multi-agent system.",
                replan_prompt,
                max_tokens=420,
            ).strip()
            new_steps = _extract_steps(new_plan)
            if not new_plan or not new_steps:
                _log(
                    "WARN",
                    "Replan",
                    "Replan output is empty or invalid. Keeping existing plan.",
                    self.settings.verbose,
                )
                ctx.post("PlanningAgent", "AllAgents", "Replan skipped due to invalid replan output.")
                return

            previous_plan = ctx.plan
            ctx.plan = new_plan
            ctx.plan_steps = new_steps
            ctx.blackboard["artifacts"]["plan"] = ctx.plan
            replans = ctx.blackboard["artifacts"].setdefault("replans", [])
            replans.append(
                {
                    "trigger_stage": stage_name,
                    "error": error_text,
                    "old_plan_preview": previous_plan[:250] if previous_plan else "",
                    "new_plan_preview": new_plan[:250],
                    "time": str(time.time()),
                }
            )
            ctx.post(
                "PlanningAgent",
                "AllAgents",
                f"Replan applied after error in {stage_name}.",
            )
            _log(
                "INFO",
                "Replan",
                f"Replan success with {len(new_steps)} step(s) after {stage_name}.",
                self.settings.verbose,
            )
        except Exception as exc:
            replan_error = f"{type(exc).__name__}: {exc}"
            ctx.warnings.append(f"Replan failed after {stage_name}: {replan_error}")
            ctx.post("Orchestrator", "AllAgents", f"Replan failed after {stage_name}: {replan_error}")
            _log("ERROR", "Replan", f"Replan failed: {replan_error}", self.settings.verbose)

    def _run_stage(
        self,
        ctx: AgentContext,
        stage_name: str,
        fn: Callable[[], None],
        replan_on_error: bool = True,
    ) -> bool:
        _log("INFO", "Orchestrator", f"START stage: {stage_name}", self.settings.verbose)
        try:
            fn()
            ctx.post("Orchestrator", stage_name, "SUCCESS")
            ctx.executed_stages.append(stage_name)
            _log("INFO", "Orchestrator", f"SUCCESS stage: {stage_name}", self.settings.verbose)
            return True
        except Exception as exc:
            error_text = self._record_error(ctx, stage_name, exc)
            if replan_on_error:
                self._replan_after_error(ctx, stage_name, error_text)
            return False

    @staticmethod
    def _default_plan_for_recovery(query: str) -> str:
        return (
            "1. Re-evaluate user objective and constraints.\n"
            "2. Collect or refresh critical evidence.\n"
            "3. Verify evidence quality and consistency.\n"
            "4. Synthesize final response with assumptions and checks.\n"
            f"Task focus: {query[:180]}"
        )

    @staticmethod
    def _fallback_answer_from_context(ctx: AgentContext) -> str:
        observations = "\n".join([f"- {obs[:220]}" for obs in ctx.observations[-4:]]) or "- No observations"
        errors = "\n".join([f"- {item.get('stage', '')}: {item.get('error', '')}" for item in ctx.errors[-5:]])
        return (
            "The architecture encountered runtime errors and switched to fallback response mode.\n\n"
            "Latest observations:\n"
            f"{observations}\n\n"
            "Recent errors:\n"
            f"{errors if errors else '- No explicit runtime errors captured.'}"
        )

    def _generate_synthesis(self, ctx: AgentContext) -> None:
        ctx.final_answer = self._synthesize(ctx)

    @staticmethod
    def _build_clarification_response(ctx: AgentContext) -> str:
        questions = ctx.clarification_questions or [
            "What exact output do you need?",
            "What inputs/files/environment should I use?",
            "What does success look like for this task?",
        ]
        missing_info = ctx.intent_analysis.get("missing_information", [])
        missing_text = "\n".join([f"- {item}" for item in missing_info[:6]]) if missing_info else "- None specified"
        question_text = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions[:5])])
        return (
            "I need a bit more context before executing the task safely.\n\n"
            f"Detected primary intent: {ctx.primary_task}\n"
            f"Ambiguity score: {ctx.ambiguity_score}\n\n"
            "Missing information:\n"
            f"{missing_text}\n\n"
            "Please clarify:\n"
            f"{question_text}"
        )

    def run(self, query: str, max_steps: int = 4) -> AgentContext:
        ctx = AgentContext(query=query, max_steps=max_steps)
        _log("INFO", "Orchestrator", f"Run started for query: {query[:220]}", self.settings.verbose)

        try:
            self._run_stage(
                ctx,
                "EpisodicSemanticAgent.recall.context",
                lambda: self.episodic_semantic.recall(ctx, self.memory),
                replan_on_error=False,
            )
            self._run_stage(
                ctx,
                "IntentParsingAgent",
                lambda: self.intent_parser.run(ctx),
                replan_on_error=False,
            )
            self._run_stage(
                ctx,
                "MetaControllerAgent",
                lambda: self.meta_controller.run(ctx),
                replan_on_error=False,
            )
            self._run_stage(
                ctx,
                "ExecutionProfile",
                lambda: self._select_execution_profile(ctx),
                replan_on_error=False,
            )
            profile = ctx.execution_profile or {}

            if bool(profile.get("run_clarification", False)):
                ctx.final_answer = self._build_clarification_response(ctx)
                ctx.post("Orchestrator", "User", "Clarification requested before full execution.")
                self._run_or_skip(
                    ctx,
                    "ReflexiveMetacognitiveAgent",
                    bool(profile.get("run_reflexive", True)),
                    lambda: self.reflexive.run(ctx),
                    "Reflexive confidence step disabled in profile.",
                    replan_on_error=False,
                )
                self._run_stage(
                    ctx,
                    "FinalizeOutput",
                    lambda: self._finalize(ctx),
                    replan_on_error=False,
                )
                self._run_stage(
                    ctx,
                    "EpisodicSemanticAgent.write",
                    lambda: self.episodic_semantic.write(ctx, self.memory),
                    replan_on_error=False,
                )
                _log(
                    "INFO",
                    "Orchestrator",
                    "Run ended in clarification mode due to high ambiguity.",
                    self.settings.verbose,
                )
                return ctx

            self._run_stage(
                ctx,
                "PlanningAgent",
                lambda: self.planning.run(ctx),
                replan_on_error=False,
            )
            if not ctx.plan.strip():
                ctx.plan = self._default_plan_for_recovery(ctx.query)
                ctx.plan_steps = _extract_steps(ctx.plan)
                ctx.warnings.append("Planning was empty. Applied default recovery plan.")
                ctx.post("Orchestrator", "AllAgents", "Default recovery plan applied.")
                _log(
                    "WARN",
                    "Orchestrator",
                    "Planning was empty. Default recovery plan inserted.",
                    self.settings.verbose,
                )

            self._run_or_skip(
                ctx,
                "GoalDecompositionAgent",
                bool(profile.get("run_goal_decomposition", False)),
                lambda: self.goal_decomposition.run(ctx),
                "Goal decomposition skipped for lightweight profile.",
            )

            self._run_or_skip(
                ctx,
                "TreeOfThoughtsAgent",
                bool(profile.get("run_tree_of_thoughts", False)),
                lambda: self.tree_of_thoughts.run(ctx),
                "Complexity/profile does not require Tree-of-Thoughts.",
            )
            self._run_or_skip(
                ctx,
                "CellularAutomataAgent",
                bool(profile.get("run_cellular_automata", False)),
                lambda: self.cellular_automata.run(ctx),
                "Complexity/profile does not require cellular prioritization.",
            )
            self._run_or_skip(
                ctx,
                "MultiAgentCoordinator",
                bool(profile.get("run_multi_agent", False)),
                lambda: self.multi_agent.run(ctx),
                "Simple task: no heavy multi-agent coordination required.",
            )
            self._run_or_skip(
                ctx,
                "ToolUseAgent",
                bool(profile.get("run_tool_use", False)),
                lambda: self.tool_use.run(ctx),
                "No external web/tool retrieval required for this task.",
            )
            self._run_or_skip(
                ctx,
                "ReActAgent",
                bool(profile.get("run_react", False)),
                lambda: self.react.run(ctx),
                "ReAct loop skipped for simple/direct execution profile.",
            )
            self._run_or_skip(
                ctx,
                "BlackboardAgent",
                bool(profile.get("run_blackboard", False)),
                lambda: self.blackboard.run(ctx),
                "Blackboard sync skipped by lightweight profile.",
            )

            run_graph = bool(profile.get("run_graph", False))
            run_ensemble = bool(profile.get("run_ensemble", False))
            if bool(profile.get("parallel_analysis", False)) and run_graph and run_ensemble:
                self._run_parallel_group(
                    ctx,
                    "analysis_branch",
                    [
                        ("GraphAgent", lambda: self.graph.run(ctx)),
                        ("EnsembleAgent", lambda: self.ensemble.run(ctx)),
                    ],
                    replan_on_error=True,
                )
            else:
                self._run_or_skip(
                    ctx,
                    "GraphAgent",
                    run_graph,
                    lambda: self.graph.run(ctx),
                    "Graph reasoning not needed for current complexity/profile.",
                )
                self._run_or_skip(
                    ctx,
                    "EnsembleAgent",
                    run_ensemble,
                    lambda: self.ensemble.run(ctx),
                    "Ensemble debate not needed for current complexity/profile.",
                )

            self._run_or_skip(
                ctx,
                "MentalLoopAgent",
                bool(profile.get("run_mental_loop", False)),
                lambda: self.mental_loop.run(ctx),
                "Mental simulation skipped by profile.",
            )
            self._run_or_skip(
                ctx,
                "DryRunAgent",
                bool(profile.get("run_dry_run", False)),
                lambda: self.dry_run.run(ctx),
                "Dry-run validation skipped by profile.",
            )
            self._run_or_skip(
                ctx,
                "PEVAgent",
                bool(profile.get("run_pev", False)),
                lambda: self.pev.run(ctx),
                "PEV verification skipped by profile.",
            )

            if (
                bool(profile.get("allow_retry_evidence", False))
                and not ctx.pev_verified
                and bool(profile.get("run_tool_use", False))
                and ctx.route.get("use_web", False)
            ):
                ctx.post("PEVAgent", "ToolUseAgent", "Verification low. Trigger one extra evidence pass.")
                _log(
                    "WARN",
                    "Orchestrator",
                    "PEV verification low. Running additional ToolUse/ReAct/Verify cycle.",
                    self.settings.verbose,
                )
                self._run_stage(ctx, "ToolUseAgent.retry", lambda: self.tool_use.run(ctx))
                if bool(profile.get("run_react", False)):
                    self._run_stage(ctx, "ReActAgent.retry", lambda: self.react.run(ctx))
                if bool(profile.get("run_blackboard", False)):
                    self._run_stage(ctx, "BlackboardAgent.retry", lambda: self.blackboard.run(ctx))
                if bool(profile.get("run_pev", False)):
                    self._run_stage(ctx, "PEVAgent.retry", lambda: self.pev.run(ctx))

            synth_ok = self._run_stage(ctx, "SynthesisEngine", lambda: self._generate_synthesis(ctx))
            if not synth_ok or not ctx.final_answer.strip():
                ctx.final_answer = self._fallback_answer_from_context(ctx)
                ctx.post("Orchestrator", "RLHFAgent", "Using fallback answer due to synthesis failure.")
                _log(
                    "WARN",
                    "Orchestrator",
                    "Fallback answer used because synthesis failed or returned empty output.",
                    self.settings.verbose,
                )

            self._run_or_skip(
                ctx,
                "RLHFAgent",
                bool(profile.get("run_rlhf", False)),
                lambda: self.rlhf.run(ctx),
                "RLHF refinement skipped for lightweight profile.",
            )
            self._run_or_skip(
                ctx,
                "ReflexiveMetacognitiveAgent",
                bool(profile.get("run_reflexive", True)),
                lambda: self.reflexive.run(ctx),
                "Reflexive confidence step disabled in profile.",
                replan_on_error=False,
            )
            self._run_stage(
                ctx,
                "FinalizeOutput",
                lambda: self._finalize(ctx),
                replan_on_error=False,
            )
            self._run_stage(
                ctx,
                "EpisodicSemanticAgent.write",
                lambda: self.episodic_semantic.write(ctx, self.memory),
                replan_on_error=False,
            )
        except Exception as exc:
            self._record_error(ctx, "Orchestrator.run", exc)
            if not ctx.final_answer:
                ctx.final_answer = self._fallback_answer_from_context(ctx)
            try:
                self._finalize(ctx)
            except Exception as finalize_exc:
                _log(
                    "ERROR",
                    "Orchestrator",
                    f"Finalize failed after orchestrator error: {type(finalize_exc).__name__}: {finalize_exc}",
                    self.settings.verbose,
                )

        _log(
            "INFO",
            "Orchestrator",
            f"Run completed. warnings={len(ctx.warnings)}, errors={len(ctx.errors)}, confidence={ctx.confidence}",
            self.settings.verbose,
        )
        return ctx

    def _synthesize(self, ctx: AgentContext) -> str:
        evidence = "\n\n".join(ctx.observations[-6:]) if ctx.observations else "No observations."
        entities = ", ".join([item.get("value", "") for item in ctx.extracted_entities[:10] if item.get("value")])
        prompt = (
            f"User query:\n{ctx.query}\n\n"
            f"Primary task: {ctx.primary_task}\n"
            f"Secondary tasks: {ctx.secondary_tasks}\n"
            f"Task probabilities: {json.dumps(ctx.task_probabilities, ensure_ascii=True)}\n"
            f"Detected entities: {entities if entities else 'none'}\n\n"
            f"Plan:\n{ctx.plan}\n\n"
            f"Assignments:\n{json.dumps(ctx.assignments, ensure_ascii=True)}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Ensemble views:\n{json.dumps(ctx.ensemble_views, ensure_ascii=True)}\n\n"
            "Write a grounded answer. Mention assumptions if evidence is weak."
        )
        return self.models.generate(
            "rag",
            "You are the final synthesis agent for an advanced multi-agent architecture.",
            prompt,
            max_tokens=760,
        ).strip()

    @staticmethod
    def _finalize(ctx: AgentContext) -> None:
        sources = _dedupe_sources(ctx.sources)
        source_text = "\n".join(
            [f"{i + 1}. {item.get('title', 'Untitled')} - {item.get('url', '')}" for i, item in enumerate(sources[:8])]
        )
        warning_text = "\n".join([f"- {w}" for w in ctx.warnings[:5]])
        error_text = "\n".join(
            [f"- {item.get('stage', '')}: {item.get('error', '')}" for item in ctx.errors[:5]]
        )
        capabilities_text = ", ".join(ctx.detected_capabilities[:12]) if ctx.detected_capabilities else "none"
        intent_tags_text = ", ".join(ctx.intent_tags[:8]) if ctx.intent_tags else "none"
        secondary_text = ", ".join(ctx.secondary_tasks[:5]) if ctx.secondary_tasks else "none"
        subtasks_text = "\n".join([f"- {task}" for task in ctx.subtasks[:8]])
        meta = (
            "\n\n---\n"
            f"Task category: {ctx.task_type}\n"
            f"Primary task: {ctx.primary_task}\n"
            f"Secondary tasks: {secondary_text}\n"
            f"Complexity: {ctx.complexity}\n"
            f"Depth: {ctx.depth}\n"
            f"Intent tags: {intent_tags_text}\n"
            f"Detected capabilities: {capabilities_text}\n"
            f"Urgency: {ctx.urgency}\n"
            f"Ambiguity score: {ctx.ambiguity_score}\n"
            f"Requires clarification: {ctx.requires_clarification}\n"
            f"Confidence: {ctx.confidence}\n"
            f"PEV verified: {ctx.pev_verified}\n"
            f"Dry-run passed: {ctx.dry_run_passed}\n"
            f"Executed stages: {len(ctx.executed_stages)}\n"
            f"Skipped stages: {len(ctx.skipped_stages)}\n"
        )
        if subtasks_text:
            meta += f"Subtasks:\n{subtasks_text}\n"
        if source_text:
            meta += f"Sources:\n{source_text}\n"
        if warning_text:
            meta += f"Warnings:\n{warning_text}\n"
        if error_text:
            meta += f"Errors:\n{error_text}\n"
        ctx.final_answer = (ctx.final_answer or "").strip() + meta


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced AutoBot architecture with full agentic flow in one main file."
    )
    parser.add_argument("--query", required=True, help="Task/query for the architecture.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum ReAct loop steps.")
    parser.add_argument("--show-trace", action="store_true", help="Print inter-agent communication trace.")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        _log("INFO", "CLI", "Loaded settings from environment.", settings.verbose)
        bot = AdvancedAutoBotArchitecture(settings)
        ctx = bot.run(query=args.query, max_steps=args.max_steps)

        print("\n=== FINAL ANSWER ===\n")
        print(ctx.final_answer)

        if args.show_trace:
            print("\n=== AGENT COMMUNICATION TRACE ===\n")
            for line in ctx.trace:
                print(f"- {line}")

        print("\n=== EXECUTION PROFILE ===\n")
        print(json.dumps(ctx.execution_profile, ensure_ascii=True, indent=2))

        if ctx.intent_analysis:
            print("\n=== INTENT ANALYSIS ===\n")
            print(json.dumps(ctx.intent_analysis, ensure_ascii=True, indent=2))

        if ctx.detected_capabilities:
            print("\n=== DETECTED CAPABILITIES ===\n")
            for capability in ctx.detected_capabilities:
                print(f"- {capability}")

        if ctx.subtasks:
            print("\n=== SUBTASKS ===\n")
            for i, task in enumerate(ctx.subtasks, start=1):
                print(f"{i}. {task}")

        print("\n=== EXECUTED STAGES ===\n")
        if ctx.executed_stages:
            print(" -> ".join(ctx.executed_stages))
        else:
            print("No stages executed.")

        if ctx.skipped_stages:
            print("\n=== SKIPPED STAGES ===\n")
            print(" -> ".join(ctx.skipped_stages))

        if ctx.requires_clarification and ctx.clarification_questions:
            print("\n=== CLARIFICATION QUESTIONS ===\n")
            for i, question in enumerate(ctx.clarification_questions, start=1):
                print(f"{i}. {question}")
    except Exception as exc:
        _log("ERROR", "CLI", f"Fatal error: {type(exc).__name__}: {exc}")
        print("\n=== FATAL ERROR ===\n")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_cli()
