import os
import time
import json
import logging
from typing import List, Dict, Optional, Tuple

# Remove ALL_PROXY before importing langchain_openai/openai so that the httpx
# clients they create internally never attempt a SOCKS5 connection (which
# requires the optional 'socksio' package).  HTTP_PROXY / HTTPS_PROXY remain
# intact so Clash can still route traffic normally.
for _k in list(os.environ):
    if 'ALL_PROXY' in _k.upper():
        os.environ.pop(_k)

from langchain_openai import ChatOpenAI
logger = logging.getLogger(__name__)

class OpenAICompatibleClient(ChatOpenAI):
    def __init__(
        self,
        model=None,
        base_url=None,
        api_key=None,
        **kwargs
    ):
        use_model = os.environ.get("LLM_MODEL", model)
        use_base_url = os.environ.get("LLM_BASE_URL", base_url)
        use_api_key = os.environ.get("LLM_API_KEY", api_key)

        super().__init__(
            model=use_model,
            base_url=use_base_url,
            api_key=use_api_key,
            **kwargs
        )

class LLMLabelGenerator:
    """LLM-based label generator for novel classes."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Init LLM label generator."""
        try:
            self.client = OpenAICompatibleClient(
                temperature=0,
                timeout=120,
                max_retries=0
            )
            logger.info(f"Initialized LLM client")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}. LLM labeling will be disabled.")
            self.client = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _create_prompt_select_indices_for_label(self, representative_texts: List[str], target_label: str, known_labels: List[str]) -> str:
        """Create prompt for selecting sample indices matching given label (strict JSON output)."""
        known_labels_str = ", ".join(known_labels)

        numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(representative_texts)])

        prompt = f"""
You are given ONE known label and a list of numbered user queries.

Goal: select only the queries whose primary intent truly matches the known label.

Rules:
- One-intent focus: focus on the main intent of each query, not other aspects of the query. Semantic match only (not keyword/substring match).
- Treat off-topic or rare mentions as outliers; EXCLUDE them.
- Boundary with OTHER LABELS: if a query fits any OTHER LABEL better, EXCLUDE it.
- Do not invent any new label. No explanation.

Note: quality over quantity; returning [] is acceptable !

KNOWN LABEL:
{target_label}

OTHER LABELS (for boundary reference):
{known_labels_str}

Output (strict JSON only; no code fences; no extra text):
{{"selected_indices": [i1, i2, ...]}}

QUERIES (numbered):
{numbered}
"""
        return prompt

    def select_indices_for_known_label(self,
                                       representative_texts: List[str],
                                       target_label: str,
                                       known_labels: List[str]) -> List[int]:
        """LLM selects 1-based indices of samples matching the given label."""
        if not self.client:
            return list(range(1, min(len(representative_texts), 60) + 1))

        prompt = self._create_prompt_select_indices_for_label(
            representative_texts, target_label, known_labels
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        import concurrent.futures

        attempt = 0
        while True:
            try:
                messages = [
                    SystemMessage(content="You are an expert at judging whether a user query belongs to a given category label. You should make precise selections."),
                    HumanMessage(content=prompt)
                ]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.invoke, messages)
                    response = future.result(timeout=120)

                raw = response.content.strip()
                indices = self._parse_indices(raw)
                if attempt > 0:
                    logger.info(f"select_indices_for_known_label succeeded after {attempt} retries")
                return indices
            except Exception as e:
                attempt += 1
                delay = min(self.retry_delay * (2 ** (attempt - 1)), 30.0)
                err_name = e.__class__.__name__
                err_msg = str(e) if str(e) else repr(e)
                logger.warning(f"select_indices_for_known_label error (attempt {attempt}): {err_name}: {err_msg}. Retrying in {delay:.2f}s")
                time.sleep(delay)


    def _create_prompt_generate_label_for_texts(self, sample_texts: List[str], known_labels: List[str]) -> str:
        """Create prompt for generating label only (no indices; strict JSON output)."""
        known_labels_str = ", ".join(known_labels)
        numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(sample_texts)])

        prompt = f"""
Context: Generalized Category Discovery / New Intent Recognition. Some intents are already known.
You will propose ONE new intent label for the samples below.

Make the label:
- One intent only: the label must describe a single coherent intent; do NOT combine multiple intents into one label.
- Handle outliers: base the label on the strongest common intent across samples; ignore off-topic or rare outlier samples.
- Form: lowercase words joined by underscores; <=5 words; concise and label-like.
- Style: express a single, specific action or outcome (one intent only); reject broad or umbrella expressions.
- Avoid generic buckets or suffixes: issues/problem/error/help/etc.
- Do NOT duplicate or be synonymous with any KNOWN LABEL; if close, add a clear qualifier (object/channel/phase/state).
- Follow the name style of KNOWN LABELS without copying any.

Output (strict JSON only; no code fences; no extra text):
{{"label": "<label_text>"}}

KNOWN LABELS (for conflict/style reference):
{known_labels_str}

SAMPLES (numbered):
{numbered}
"""
        return prompt

    def generate_label_for_texts(self,
                                 sample_texts: List[str],
                                 known_labels: List[str],
                                 cluster_id: int = None) -> Optional[str]:
        """Generate one label (no indices); avoid duplicating known labels."""
        if not self.client:
            raise Exception("LLM client not available.")

        prompt = self._create_prompt_generate_label_for_texts(sample_texts, known_labels)

        from langchain_core.messages import HumanMessage, SystemMessage
        import concurrent.futures

        attempt = 0
        while True:
            try:
                messages = [
                    SystemMessage(content="You are an expert at categorizing text data and creating concise, meaningful labels."),
                    HumanMessage(content=prompt)
                ]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.invoke, messages)
                    response = future.result(timeout=120)

                raw = response.content.strip()
                import json
                import re
                label = None
                text = raw.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                        text = "\n".join(lines[1:-1]).strip()
                else:
                    text = raw

                try:
                    data = json.loads(text)
                    label = self._clean_and_validate_label(str(data.get("label", "")))
                except Exception:
                    pass

                if not label:
                    m = re.search(r"\"label\"\s*:\s*\"([^\"]+)\"", text)
                    if m:
                        label = self._clean_and_validate_label(m.group(1))

                if not label:
                    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("```")]
                    first_line = lines[0].strip() if lines else ""
                    m2 = re.search(r"label\s*[:：]\s*(.+)", first_line, flags=re.I)
                    label = (m2.group(1).strip() if m2 else first_line) or None
                    label = self._clean_and_validate_label(label)

                if label:
                    if attempt > 0:
                        logger.info(f"generate_label_for_texts succeeded after {attempt} retries for cluster {cluster_id}")
                    logger.info(f"Generated label for cluster {cluster_id}: {label}")
                    return label
                else:
                    raise ValueError("Parsed empty label from model response")
            except Exception as e:
                attempt += 1
                delay = min(self.retry_delay * (2 ** (attempt - 1)), 30.0)
                err_name = e.__class__.__name__
                err_msg = str(e) if str(e) else repr(e)
                logger.error(f"Error generating label for cluster {cluster_id}, attempt {attempt}: {err_name}: {err_msg}")
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)


    def generate_labels_for_groups_mixed(
        self,
        groups_sample_texts: List[List[str]],
        forbidden_labels_local: List[str],
        global_forbidden_labels: List[str],
        known_labels_context: List[Dict],
        debug_log: bool = False,
        debug_dump_prefix: Optional[str] = None
    ) -> Optional[List[str]]:
        """Mixed component: generate labels for novel groups; avoid local+global forbidden; no post-processing."""
        if not self.client:
            raise Exception("LLM client not available.")

        if not groups_sample_texts:
            return []

        G = len(groups_sample_texts)

        try:
            logger.info(
                f"[LLM-GROUP-LABELS-MIXED] start: groups={G}, "
                f"forbidden_local={len(forbidden_labels_local or [])}, "
                f"forbidden_global={len(global_forbidden_labels or [])}, "
                f"known_ctx={len(known_labels_context or [])}"
            )
        except Exception:
            pass

        known_blocks = []
        try:
            for item in (known_labels_context or []):
                lab = item.get('label', '')
                ktexts = item.get('sample_texts', []) or []
                numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(ktexts)])
                block = f"LABEL: {lab}\nSAMPLES of this label:\n{numbered}\n"
                known_blocks.append(block)
        except Exception:
            pass
        known_joined = "\n".join(known_blocks)

        forbidden_local_str = ", ".join(forbidden_labels_local or [])
        forbidden_global_str = ", ".join(global_forbidden_labels or [])
        style_samples = (global_forbidden_labels or [])[:12]
        style_examples_str = ", ".join(style_samples)

        groups_blocks = []
        for gi, texts in enumerate(groups_sample_texts, start=1):
            numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts or [])])
            block = f"Group {gi}:\n{numbered}\n"
            groups_blocks.append(block)
        groups_joined = "\n".join(groups_blocks)

        prompt = f"""
Context: You are given {G} groups of user queries. Each group needs ONE distinct intent label.

IMPORTANT: These {G} groups are semantically SIMILAR to FORBIDDEN-LOCAL labels below (clustered together due to high similarity). You must find subtle differences and add strong qualifiers (object/channel/phase/state) to distinguish them.

Task: Propose EXACTLY {G} labels.

Constraints:
- Pairwise distinctiveness: the {G} labels MUST be mutually exclusive; no synonyms or paraphrases among outputs.
- FORBIDDEN-LOCAL (semantically close): do NOT duplicate or be synonymous; study sample texts to understand boundaries.
- GLOBAL-FORBIDDEN: also avoid duplications.
- Form: lowercase words joined by underscores; <=5 words; concise and label-like.
- Style: follow GLOBAL-FORBIDDEN naming style; avoid generic suffixes (issue/problem/error/help).

Output (strict JSON only; no code fences; no extra text):
{{"labels": ["<label_1>", "<label_2>", ..., "<label_{G}>"]}}

FORBIDDEN-LOCAL (semantically SIMILAR; must distinguish):
{forbidden_local_str}

{known_joined}

GLOBAL-FORBIDDEN:
{forbidden_global_str}

GROUPS:
{groups_joined}
"""

        if debug_log:
            try:
                logger.info("[FORCE-RENAME DEBUG][MIXED] Prompt:\n" + prompt)
                if debug_dump_prefix:
                    try:
                        os.makedirs(os.path.dirname(debug_dump_prefix), exist_ok=True)
                        with open(debug_dump_prefix + ".prompt.txt", "w", encoding="utf-8") as f:
                            f.write(prompt)
                    except Exception as _:
                        pass
            except Exception:
                pass

        from langchain_core.messages import HumanMessage, SystemMessage
        import concurrent.futures, json, re

        def _parse_labels_array(raw: str) -> Optional[List[str]]:
            text = raw.strip()
            # strip code fences if any
            if text.startswith("```") and text.endswith("```"):
                lines = text.splitlines()
                if len(lines) >= 3:
                    text = "\n".join(lines[1:-1]).strip()
            # strict json
            try:
                data = json.loads(text)
                arr = data.get("labels", None)
                if isinstance(arr, list):
                    return [self._clean_and_validate_label(str(x)) for x in arr]
            except Exception:
                pass
            # fallback: try to find first [...] and parse
            m = re.search(r"\[(.*?)\]", text, flags=re.S)
            if m:
                try:
                    arr_text = "[" + m.group(1) + "]"
                    arr = json.loads(arr_text)
                    if isinstance(arr, list):
                        return [self._clean_and_validate_label(str(x)) for x in arr]
                except Exception:
                    pass
            return None

        attempt = 0
        while True:
            attempt += 1
            try:
                messages = [
                    SystemMessage(content="You answer only with minimal strict JSON."),
                    HumanMessage(content=prompt)
                ]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.invoke, messages)
                    response = future.result(timeout=120)

                raw = response.content.strip()
                if debug_log:
                    try:
                        _r = raw
                        if _r.startswith("```") and _r.endswith("```"):
                            _lines = _r.splitlines()
                            if len(_lines) >= 3:
                                _r = "\n".join(_lines[1:-1]).strip()
                        logger.info("[FORCE-RENAME DEBUG][MIXED] Raw response:\n" + _r)
                        if debug_dump_prefix:
                            try:
                                with open(debug_dump_prefix + ".response.txt", "w", encoding="utf-8") as f:
                                    f.write(_r)
                            except Exception as _:
                                pass
                    except Exception:
                        pass
                labels = _parse_labels_array(raw)
                if labels and len(labels) == G:
                    try:
                        if attempt > 1:
                            logger.info(f"[LLM-GROUP-LABELS-MIXED] succeeded after {attempt-1} retries")
                        preview = labels if len(labels) <= 5 else labels[:5] + ["..."]
                        logger.info(f"[LLM-GROUP-LABELS-MIXED] labels={preview}")
                    except Exception:
                        pass
                    return labels

                delay = min(self.retry_delay * (2 ** (attempt - 1)), 30.0)
                logger.warning(f"[LLM-GROUP-LABELS-MIXED][PARSE-RETRY {attempt}] invalid/empty/len!=G. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            except Exception as e:
                delay = min(self.retry_delay * (2 ** (attempt - 1)), 30.0)
                logger.warning(f"[LLM-GROUP-LABELS-MIXED][API-RETRY {attempt}] error: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)


    def generate_labels_for_groups_pure(
        self,
        groups_sample_texts: List[List[str]],
        global_forbidden_labels: List[str],
        debug_log: bool = False
    ) -> Optional[List[str]]:
        """Pure novel component: generate labels for novel groups; avoid global forbidden; no post-processing."""
        if not self.client:
            raise Exception("LLM client not available.")

        if not groups_sample_texts:
            return []

        G = len(groups_sample_texts)
        try:
            logger.info(
                f"[LLM-GROUP-LABELS-PURE] start: groups={G}, "
                f"forbidden_global={len(global_forbidden_labels or [])}"
            )
        except Exception:
            pass
        forbidden_global_str = ", ".join(global_forbidden_labels or [])

        # assemble group texts
        groups_blocks = []
        for gi, texts in enumerate(groups_sample_texts, start=1):
            numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts or [])])
            block = f"Group {gi}:\n{numbered}\n"
            groups_blocks.append(block)
        groups_joined = "\n".join(groups_blocks)


        prompt = f"""
Context: You are given {G} groups of user queries. Each group needs ONE distinct intent label.

IMPORTANT: These {G} groups are semantically SIMILAR to each other (clustered together due to high similarity). You must add strong qualifiers (object/channel/phase/state) to ensure clear separation between them.

Task: Propose EXACTLY {G} labels.

Constraints:
- Pairwise distinctiveness: the {G} labels MUST be mutually exclusive; no synonyms or paraphrases among outputs.
- GLOBAL-FORBIDDEN: do NOT duplicate or be synonymous.
- Form: lowercase words joined by underscores; <=5 words; concise and label-like.
- Style: follow GLOBAL-FORBIDDEN naming style; avoid generic suffixes (issue/problem/error/help).

Output (strict JSON only; no code fences; no extra text):
{{"labels": ["<label_1>", "<label_2>", ..., "<label_{G}>"]}}

GLOBAL-FORBIDDEN:
{forbidden_global_str}

GROUPS:
{groups_joined}
"""

        if debug_log:
            try:
                logger.info("[FORCE-RENAME DEBUG][PURE] Prompt:\n" + prompt)
            except Exception:
                pass

        from langchain_core.messages import HumanMessage, SystemMessage
        import concurrent.futures, json, re

        def _parse_labels_array(raw: str) -> Optional[List[str]]:
            text = raw.strip()
            # strip code fences if any
            if text.startswith("```") and text.endswith("```"):
                lines = text.splitlines()
                if len(lines) >= 3:
                    text = "\n".join(lines[1:-1]).strip()
            # strict json
            try:
                data = json.loads(text)
                arr = data.get("labels", None)
                if isinstance(arr, list):
                    return [self._clean_and_validate_label(str(x)) for x in arr]
            except Exception:
                pass
            # fallback: try to find first [...] and parse
            m = re.search(r"\[(.*?)\]", text, flags=re.S)
            if m:
                try:
                    arr_text = "[" + m.group(1) + "]"
                    arr = json.loads(arr_text)
                    if isinstance(arr, list):
                        return [self._clean_and_validate_label(str(x)) for x in arr]
                except Exception:
                    pass
            return None

        attempt = 0
        while True:
            attempt += 1
            try:
                messages = [
                    SystemMessage(content="You answer only with minimal strict JSON."),
                    HumanMessage(content=prompt)
                ]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.invoke, messages)
                    response = future.result(timeout=120)

                raw = response.content.strip()
                if debug_log:
                    try:
                        _r = raw
                        if _r.startswith("```") and _r.endswith("```"):
                            _lines = _r.splitlines()
                            if len(_lines) >= 3:
                                _r = "\n".join(_lines[1:-1]).strip()
                        logger.info("[FORCE-RENAME DEBUG][PURE] Raw response:\n" + _r)
                    except Exception:
                        pass
                labels = _parse_labels_array(raw)
                if labels and len(labels) == G:
                    try:
                        if attempt > 1:
                            logger.info(f"[LLM-GROUP-LABELS-PURE] succeeded after {attempt-1} retries")
                        preview = labels if len(labels) <= 5 else labels[:5] + ["..."]
                        logger.info(f"[LLM-GROUP-LABELS-PURE] labels={preview}")
                    except Exception:
                        pass
                    return labels

                delay = min(self.retry_delay * (2 ** (attempt - 1)), 30.0)
                logger.warning(f"[LLM-GROUP-LABELS-PURE][PARSE-RETRY {attempt}] invalid/empty/len!=G. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            except Exception as e:
                delay = min(self.retry_delay * (2 ** (attempt - 1)), 30.0)
                logger.warning(f"[LLM-GROUP-LABELS-PURE][API-RETRY {attempt}] error: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
    def _parse_indices(self, raw: str) -> List[int]:
        """Parse indices from response (strict JSON preferred, fallback to [...] extraction)."""
        import json
        import re
        try:
            data = json.loads(raw)
            arr = data.get("selected_indices", [])
            if isinstance(arr, list):
                return [int(i) for i in arr if isinstance(i, int) or (isinstance(i, str) and str(i).isdigit())]
        except Exception:
            pass
        m = re.search(r"\[(.*?)\]", raw, flags=re.S)
        if m:
            try:
                arr_text = "[" + m.group(1) + "]"
                arr = json.loads(arr_text)
                if isinstance(arr, list):
                    return [int(i) for i in arr if isinstance(i, int) or (isinstance(i, str) and str(i).isdigit())]
            except Exception:
                nums = re.findall(r"\d+", m.group(1))
                return [int(n) for n in nums]
        return []
    
    def _clean_and_validate_label(self, label: str) -> Optional[str]:
        """Clean and validate generated label."""
        if not label:
            return None
        import re
        label = label.strip()
        label = label.strip('"\'`')
        if (label.startswith('(') and label.endswith(')')) or (label.startswith('[') and label.endswith(']')):
            label = label[1:-1].strip()
        label = re.sub(r"\s+", " ", label)
        if len(label) > 100:
            label = label[:100].rstrip()
        return label

def create_label_generator() -> LLMLabelGenerator:

    return LLMLabelGenerator()
