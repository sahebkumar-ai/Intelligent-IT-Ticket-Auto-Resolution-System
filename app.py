
import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))
from src.ticket_pipeline import TicketPipeline
from src.ocr_processor import OCRProcessor
st.set_page_config(
    page_title="IT Ticket Auto-Resolution",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        text-align: center;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_pipeline():

    return TicketPipeline(
        confidence_threshold=0.80
    )


@st.cache_resource
def load_ocr_processor():
    return OCRProcessor()


try:

    pipeline = load_pipeline()
    ocr_processor = load_ocr_processor()

except Exception as exc:

    st.error(
        "Application initialization failed."
    )

    st.code(
        str(exc)
    )

    st.info(
        "Make sure the ML model has been trained "
        "using: python src/train_model.py"
    )

    st.stop()
with st.sidebar:

    st.header(
        " System Configuration"
    )

    confidence_threshold = st.slider(
        "Automation Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01
    )

    pipeline.confidence_threshold = (
        confidence_threshold
    )

    st.divider()

    st.subheader(
        "Supported Inputs"
    )

    st.write(
        "✓ Text tickets"
    )

    st.write(
        "✓ Screenshots"
    )

    st.write(
        "✓ Error messages"
    )

    st.write(
        "✓ System logs"
    )

    st.divider()

    st.subheader(
        "System Targets"
    )

    st.write(
        "Classification Accuracy: ≥ 80%"
    )

    st.write(
        "Target Response Time: < 2 seconds"
    )

    st.write(
        "Human Escalation: Enabled"
    )
st.markdown(
    '<div class="main-title">'
    '🎫 Intelligent IT Ticket Auto-Resolution'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    
    '</div>',
    unsafe_allow_html=True
)
col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Issue Categories",
        "20+"
    )

with col2:

    st.metric(
        "Automation Threshold",
        f"{confidence_threshold:.0%}"
    )

with col3:

    st.metric(
        "Response Target",
        "< 2 sec"
    )

with col4:

    st.metric(
        "Input Formats",
        "Text + Image"
    )


st.divider()
st.markdown(
    '<div class="section-title">'
    'Submit IT Support Ticket'
    '</div>',
    unsafe_allow_html=True
)

ticket_text = st.text_area(
    "Describe the issue",
    placeholder=(
        "Example: My VPN keeps disconnecting "
        "when I try to connect to the company network."
    ),
    height=150
)
uploaded_image = st.file_uploader(
    "Upload a screenshot (optional)",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tiff"
    ],
    help=(
        "Upload a screenshot containing an "
        "error message or system information."
    )
)
ocr_text = ""

if uploaded_image:

    st.markdown(
        '<div class="section-title">'
        ' Screenshot Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    preview_col, text_col = st.columns(
        [1, 1]
    )

    with preview_col:

        st.image(
            uploaded_image,
            caption="Uploaded Screenshot",
            use_column_width=True
        )

    with text_col:

        with st.spinner(
            "Extracting text from screenshot..."
        ):

            try:

                ocr_text = (
                    ocr_processor
                    .extract_from_uploaded_file(
                        uploaded_image
                    )
                )

            except Exception as exc:

                st.warning(
                    "OCR processing could not be completed."
                )

                st.caption(
                    str(exc)
                )

        if ocr_text:

            st.text_area(
                "Extracted OCR Text",
                value=ocr_text,
                height=180
            )

        else:

            st.info(
                "No readable text was detected "
                "in the screenshot."
            )

with st.expander(
    " Add system/application logs (optional)"
):

    log_text = st.text_area(
        "Paste logs here",
        placeholder=(
            "Example:\n"
            "ERROR 503 Service Unavailable\n"
            "Database connection timeout\n"
            "Connection refused"
        ),
        height=150
    )
st.divider()

analyze_button = st.button(
    "🔍 Analyze & Recommend Solution",
    type="primary",
    use_container_width=True
)
if analyze_button:

    if not ticket_text.strip() and not ocr_text.strip():

        st.warning(
            "Please enter a ticket description "
            "or upload a screenshot."
        )

        st.stop()

    with st.spinner(
        "Analyzing IT ticket..."
    ):

        try:

            result = pipeline.process_ticket(
                ticket_text=ticket_text,
                ocr_text=ocr_text,
                log_text=log_text,
                ticket_id="WEB-DEMO"
            )

        except Exception as exc:

            st.error(
                "Ticket processing failed."
            )

            st.exception(
                exc
            )

            st.stop()
    st.success(
        "Ticket analysis completed successfully."
    )

    classification = result[
        "classification"
    ]

    resolution = result[
        "resolution"
    ]

    performance = result[
        "performance"
    ]
    st.markdown(
        '<div class="section-title">'
        'Classification Result'
        '</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        st.metric(
            "Issue Category",
            classification[
                "category"
            ]
        )

    with result_col2:

        st.metric(
            "Confidence",
            f"{classification['confidence']:.1%}"
        )

    with result_col3:

        st.metric(
            "Processing Time",
            f"{performance['processing_time_seconds']:.3f}s"
        )
    st.markdown(
        '<div class="section-title">'
        ' Resolution Decision'
        '</div>',
        unsafe_allow_html=True
    )

    if resolution[
        "requires_human_review"
    ]:

        st.warning(
            " Human Review Required"
        )

        st.write(
            "The model confidence is below the "
            f"{confidence_threshold:.0%} automation threshold."
        )

    else:

        st.success(
            " Automated Recommendation Available"
        )

        st.write(
            "The model confidence meets the "
            f"{confidence_threshold:.0%} automation threshold."
        )
    st.markdown(
        '<div class="section-title">'
        ' Recommended Solution'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        resolution[
            "solution"
        ]
    )

    st.markdown(
        "# Recommended Troubleshooting Steps"
    )

    steps = resolution[
        "steps"
    ]

    for index, step in enumerate(
        steps,
        start=1
    ):

        st.write(
            f"**{index}.** {step}"
        )
    st.markdown(
        "###  Performance"
    )

    processing_time = performance[
        "processing_time_seconds"
    ]

    if performance[
        "within_target"
    ]:

        st.success(
            f"Response generated in "
            f"{processing_time:.3f} seconds "
            f"(target < 2 seconds)."
        )

    else:

        st.warning(
            f"Response generated in "
            f"{processing_time:.3f} seconds, "
            "which exceeds the current target."
        )
    with st.expander(
        " View Technical Details"
    ):

        st.json(
            result
        )
st.divider()

st.caption(
    "Intelligent IT Ticket Auto-Resolution System "
    
)
