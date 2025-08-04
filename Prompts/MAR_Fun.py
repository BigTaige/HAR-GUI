Memory_JUDGE_PROMPT = """
The following is the output of an agent performing an GUI episodic reasoning task in an environment, representing the reflection and thinking process of the agent during task execution. 

Below are the actions that the agent has previously completed:
<history>(history)</history>

Below is the agent's output:
(pred)

***Your task***
Since historical interaction records (what the agent has done before) are the key clues for the agent to make correct sequence predictions, you need to determine whether the content of the "<think> ... </think>" tag in the agent's output directly contains the previously completed interaction information contained in the "<history> ... </history>" tag. 
If it has been considered, output "yes" and the reason; otherwise, output "no"."""


