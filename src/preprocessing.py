
import re
import unicodedata
def normalize_text(text: str) -> str:
    
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    return text


def clean_text(text: str) -> str:
    
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

   
    text = normalize_text(text)

    text = text.lower()

   
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

   
    text = re.sub(
        r"[_/\\|]+",
        " ",
        text
    )

    
    text = re.sub(
        r"[^a-z0-9\s.,!?'-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def preprocess_ticket(text: str) -> str:
    

    cleaned = clean_text(text)

   
    if len(cleaned.strip()) < 3:
        return ""

    return cleaned


def combine_ticket_text(
    ticket_text: str,
    ocr_text: str = "",
    log_text: str = ""
) -> str:
    
    parts = []

    if ticket_text:
        parts.append(str(ticket_text))

    if ocr_text:
        parts.append(str(ocr_text))

    if log_text:
        parts.append(str(log_text))

    combined_text = " ".join(parts)

    return preprocess_ticket(combined_text)


def is_valid_ticket(text: str) -> bool:
    
    processed_text = preprocess_ticket(text)

    return len(processed_text) >= 3


if __name__ == "__main__":
    # Simple local test
    sample_ticket = """
        VPN NOT WORKING!!!
        User cannot connect to https://company-vpn.com
        Contact: employee@example.com
    """

    print("Original Ticket:")
    print(sample_ticket)

    print("\nProcessed Ticket:")
    print(preprocess_ticket(sample_ticket))

