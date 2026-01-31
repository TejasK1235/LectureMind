from summarization.utils import load_json, save_text
from summarization.extractor import select_top_segments
from summarization.llm_summarizer import summarize_with_llm


def main():
    data = load_json("tagging2/data/tagged/tagged_output.json")

    lecture_segments = [
        s for s in data if s["tag"] == "LECTURE_CONTENT"
    ]

    selected = select_top_segments(lecture_segments, ratio=0.25)

    summary = summarize_with_llm(selected)

    save_text("summary.txt", summary)
    print("Summary saved to summary.txt")



def run_summarization_pipeline(tagged_segments):
    """
    Input: tagged segments (list of dicts)
    Output: summary string
    """
    lecture_segments = [
        s for s in tagged_segments if s["tag"] == "LECTURE_CONTENT"
    ]

    selected = select_top_segments(lecture_segments, ratio=0.30)
    summary = summarize_with_llm(selected)

    return summary



if __name__ == "__main__":
    main()

