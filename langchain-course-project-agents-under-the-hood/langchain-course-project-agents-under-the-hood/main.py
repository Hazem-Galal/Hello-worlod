from ollama import chat



def main():
    response = chat(
        model='gemma3:270m',  # Uses your installed model (see: ollama list)
        messages=[{'role': 'user', 'content': 'Hello!'}],
    )
    print(response.message.content)

if __name__ == '__main__':
    main()
