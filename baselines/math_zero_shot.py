"""No-retrieval math baseline (paper Table 6, "No Retriever")."""


class Harness:
    def run(self, task, model, trace):
        problem = str(task.get("problem", task.get("input", "")))
        prompt = ("Solve the problem rigorously. End with the final answer inside \\boxed{}.\n\n"
                  f"Problem:\n{problem}\n\nSolution:")
        trace.event("math_prompt", {"prompt": prompt})
        answer = str(model(prompt)).strip()
        trace.event("math_result", {"answer": answer})
        return answer
