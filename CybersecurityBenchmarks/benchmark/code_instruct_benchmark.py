# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import numpy as np

from pathlib import Path

from CodeShield.insecure_code_detector import insecure_code_detector
from CodeShield.insecure_code_detector.languages import Language

from tqdm import tqdm

from .benchmark import Benchmark
from .bleu import compute_bleu_score
from .query_llm import query_llm_to_generate_responses

from sentence_transformers import SentenceTransformer, util

LOG: logging.Logger = logging.getLogger(__name__)


class CodeInstructBenchmark(Benchmark):
    """
    Class Implementation for code_instruct benchmark
    """

    @classmethod
    def return_kind(cls) -> list[str]:
        return ["code_instruct"]

    def query_llm_to_generate_responses(
        self, prompt_path: Path, run_llm_in_parallel: int = 1
    ) -> None:
        # The query requires the valid prompt path
        response_result = []
        for llm in self.llms_under_test:
            response_result.extend(
                query_llm_to_generate_responses(
                    {"initial_query_llm": llm},
                    prompt_path,
                    run_llm_in_parallel,
                    num_test_cases=self.num_test_cases,
                    pass_k=self.pass_k,
                )
            )

        self.response_path.write_text(json.dumps(response_result, indent=4))

    async def run(self, num_test_cases: int = 0, run_llm_in_parallel: int = 1) -> None:  # noqa: C901
        """
        Processes a LLM-response by checking whether the response is detected as insecure by ICD.
        """
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        responses = json.loads(self.response_path.read_text())
        if num_test_cases > 0:
            responses = responses[:num_test_cases]

        results = {}

        models = {response["model"] for response in responses}
        for model in models:
            results[model] = {}

            model_responses = [
                response for response in responses if response["model"] == model
            ]

            LOG.info(f"Evaluating {model}...")

            for language in Language:
                # Tracking metrics for each language
                if not any(
                    response["language"] == language.value
                    for response in model_responses
                ):
                    LOG.info(
                        f"Skipping language {language.value} because it's not present in the dataset."
                    )
                    continue
                else:
                    LOG.info(
                        f"Starting to process responses with language {language.value}."
                    )

                lang_responses = list(
                    filter(lambda x: x["language"] == language.value, model_responses)
                )
                language_total_count = len(lang_responses)

                cosine_similarities = []
                for llm_response in tqdm(lang_responses, desc="Processing Responses"):
                    response = llm_response["response"]

                    response_embed = embed_model.encode(response, convert_to_tensor=True)
                    pattern_desc_embed = embed_model.encode(llm_response["pattern_desc"], convert_to_tensor=True)

                    cosine_similarities.append(util.cos_sim(response_embed, pattern_desc_embed)[0][0].cpu())


                results[model][language] = {
                    "total_count": language_total_count,
                    "cosine_similarities_avg": str(np.mean(cosine_similarities)),
                    "cosine_similarities_std": str(np.std(cosine_similarities)),
                }

        if (stat_path := self.stat_path) is not None:
            stat_path.write_text(json.dumps(results, indent=4))


