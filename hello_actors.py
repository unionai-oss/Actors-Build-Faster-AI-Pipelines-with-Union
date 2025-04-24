import union

actor = union.ActorEnvironment(
    name="my-actor-container", # unique name for the actor env
    replica_count=1,# Number of actor replicas to provision.
    ttl_seconds=120,  # Keep the actor alive even when idle
    requests=union.Resources(cpu="2", mem="300Mi"), # Compute resources actor requires
)
# update to just @task if you want to see the difference
@actor.task
def add_ints(num1: int, num2: int) -> int:
    return num1 + num2

@union.workflow
def wf_add()  -> int:
  num = add_ints(4, 2)
  num = add_ints(num, 5)
  num = add_ints(num, 2)
  num = add_ints(num, 4)

  return num

# once authenticated, run the workflow with:
# union run --remote hello_actors.py wf_add