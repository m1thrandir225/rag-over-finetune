import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from internal.config import ConfigLoader
from internal.rag import RAG


async def main():
    config_loader = ConfigLoader("./config.json")
    config = config_loader.load_config()

    print(f"MCP Enabled: {config.mcp_enabled}")
    print(f"MCP Servers: {list(config.mcp_servers.keys())}")
    print()

    if not config.mcp_enabled:
        print("MCP tools are not enabled")
        return

    rag = RAG(config)

    # Clear any existing documents from previous runs
    rag.clear()

    sample_texts = [
        "Македонија е држава во балканот, a нејзиниот главен град е Скопје.",
        "Македонскиот јазик има повеќе од 2 милиони говорници.",
        "Низ Скопје тече реката Вардар.",
    ]
    rag.add_texts(sample_texts)
    print(f"Added {rag.document_count()} documents")
    print()

    # Example 1: Regular query without tools
    print("-" * 10)
    print("Пример 1: Обично Прашање (Без MCP)")

    question1 = "Кој е главниот град на Македонија?"
    print(f"Прашање: {question1}")
    answer1 = rag.query_simple(question1)
    print(f"Одговор: {answer1}")
    print()

    # Example 2: Query with tools
    print("-" * 10)
    print("Пример 2: Прашање со MCP Алатки (Време)")

    question2 = "Колку е часот сега во Скопје?"
    print(f"Прашање: {question2}")
    print()

    try:
        result = await rag.query_with_tools(question2, include_scores=False)

        print(f"Одговор: {result['answer']}")
        print()

        if result["tool_calls"]:
            print(f"Tools Used: {len(result['tool_calls'])}")
            for i, tool_call in enumerate(result["tool_calls"], 1):
                print(f"  {i}. {tool_call['name']}")
                print(f"     Arguments: {tool_call['args']}")
        else:
            print("No tools were used for this question")

    except Exception as e:
        print(f"Error: {e}")

    print()

    # Example 3: Combined query
    print("-" * 10)
    print("Пример 3: Прашање за Документите")

    question3 = "Колку луѓе живеат во Скопје и колку е часот таму?"
    print(f"Прашање: {question3}")
    print()

    try:
        result = await rag.query_with_tools(question3, include_scores=True)
        print(f"Одговор: {result['answer']}")
        print()

        if result["tool_calls"]:
            print(f"Tools Used:")
            for tool_call in result["tool_calls"]:
                print(f"  - {tool_call['name']}: {tool_call['args']}")

        if result.get("sources"):
            print(f"\nSources: {len(result['sources'])} documents")

    except Exception as e:
        print(f"Error: {e}")

    print()


if __name__ == "__main__":
    asyncio.run(main())