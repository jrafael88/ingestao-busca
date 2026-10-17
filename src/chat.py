from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (ou digite 'sair' para encerrar):")

    while True:
        pergunta = input("\nPERGUNTA: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() in ("sair", "exit", "quit"):
            break

        resposta = chain(pergunta)
        print(f"RESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
