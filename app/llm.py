from google import genai

class GeminiLLM:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate(self, query, context):
        prompt = f"""
        You are a finance assistant.
        Answer ONLY using the provided context.
        If information is not available, say "Information not found."

        Context:
        {context}

        Question:
        {query}
        """

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return response.text