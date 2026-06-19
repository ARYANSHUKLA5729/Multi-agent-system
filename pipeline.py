from agents import (
    builder_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain
)

def run_research_pipeline(topic: str) -> dict:

    state = {}

    # STEP 1 - SEARCH AGENT
    print("\n" + "=" * 50)
    print("STEP 1 - Search Agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            ("user", f"Find recent, reliable and detailed information about: {topic}")
        ]
    })

    print("\n===== RAW SEARCH RESULT =====")
    print(search_result)
    print("=============================")

    state["search_results"] = search_result["messages"][-1].content

    print("\n===== SEARCH RESULT CONTENT =====")
    print(state["search_results"])
    print("=================================")

    # STEP 2 - READER AGENT
    print("\n" + "=" * 50)
    print("STEP 2 - Reader Agent is scraping top resources...")
    print("=" * 50)

    print("\n===== INPUT SENT TO READER AGENT =====")
    print(state["search_results"])
    print("======================================")

    reader_agent = builder_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Based on the following search results about '{topic}',

Pick the most relevant URL and use the scrape_url tool to scrape it.

Search Results:
{state['search_results']}
"""
            )
        ]
    })

    print("\n===== RAW READER RESULT =====")
    print(reader_result)
    print("=============================")

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\n===== SCRAPED CONTENT =====")
    print(state["scraped_content"])
    print("===========================")

    # STEP 3 - WRITER
    print("\n" + "=" * 50)
    print("STEP 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = f"""
SEARCH RESULTS:
{state['search_results']}

DETAILED SCRAPED CONTENT:
{state['scraped_content']}
"""

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\n===== FINAL REPORT =====")
    print(state["report"])
    print("========================")

    # STEP 4 - CRITIC
    print("\n" + "=" * 50)
    print("STEP 4 - Critic is reviewing the report...")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\n===== CRITIC FEEDBACK =====")
    print(state["feedback"])
    print("===========================")

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)