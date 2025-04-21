from src.LanguageModel import LanguageModel
from scipy.spatial.distance import cosine

lm = LanguageModel()


questions_file = "data/test_questions.txt"
answers_file = "data/test_answers.txt"
output_file = "data/model_outputs.txt"

questions_file = "data/questions.txt"
answers_file = "data/answers.txt"
output_file = "data/model_outputs.txt"

# lm.ask(request.question)
embeddings = lm.embeddings


# Function to compute cosine similarity
def compute_cosine_similarity(text1, text2):
    # Generate embeddings
    embed1 = embeddings.embed_query(text1)
    embed2 = embeddings.embed_query(text2)
    # Compute and return cosine similarity
    return 1 - cosine(embed1, embed2)


counter = 0
acc_sum = 0
# Main processing loop
with (
    open(questions_file, "r") as q_file,
    open(answers_file, "r") as a_file,
    open(output_file, "w") as o_file,
):
    questions = q_file.readlines()
    expected_answers = a_file.readlines()

    for i, question in enumerate(questions):
        question = question.strip()
        expected_answer = expected_answers[i].strip()

        # Get the LLM-generated answer
        generated_answer = lm.ask(question)

        # Write generated answer to output file
        o_file.write(generated_answer + "\n")

        # Compute cosine similarity
        similarity = compute_cosine_similarity(generated_answer, expected_answer)
        acc_sum = acc_sum + similarity
        counter = counter + 1
        # Print results
        print(f"Question: {question}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Cosine Similarity: {similarity:.4f}")
        print("-" * 50)

print(
    "Processing complete. Check model.output.txt for the generated answers.Avg acc. is",
    str(acc_sum / counter),
)
