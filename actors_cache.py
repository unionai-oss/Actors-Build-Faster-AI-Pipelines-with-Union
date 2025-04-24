import union
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from flytekit import ImageSpec, Resources
from union.actor import ActorEnvironment

image = ImageSpec(
    packages=[
        "union",
        "transformers",
        "torch",
        "accelerate",
    ],
    builder="union",
)

llm_actor = ActorEnvironment(
    name="gpu-llm-actor",
    container_image=image,
    replica_count=1,
    ttl_seconds=120,
    requests=Resources(cpu="1", mem="2000Mi", gpu="1"),
)


@union.actor_cache
def load_model(model_name: str = "microsoft/Phi-4-mini-instruct") -> pipeline:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


@llm_actor.task
def actor_model_predict(query: str, prev_summary: str ="nothing yet.") -> str:
    nlp_pipeline = load_model()

    # Chain query using the previous output
    full_query = (
        f"{query} Previously you told me about {prev_summary}."
        "Keep answers short and 1 sentence long."
        "Include a list of all insects we discussed after the answer."
    )

    predictions = nlp_pipeline(full_query, batch_size=1, return_full_text=False)
    return predictions[0]["generated_text"]


@union.workflow
def wf_text_gen() -> str:
    result_ant = actor_model_predict(query="What is an ant?")
    result_bee = actor_model_predict(query="What is a bee", prev_summary=result_ant)
    result_wasp = actor_model_predict(query="What is a wasp", prev_summary=result_bee)
    return result_wasp