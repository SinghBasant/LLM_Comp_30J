import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from streamlit_extras.colored_header import colored_header
from streamlit_modal import Modal
from fireworks.client import Fireworks
from together import Together

# Load environment variables --local machine
load_dotenv()

#Initialize API clients -- LOCAL MACHINE
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
fireworks_client = Fireworks(api_key=os.getenv("FIREWORKS_DS_API_KEY"))
together_client = Together(api_key=os.getenv("TOGETHER_ML_API_KEY"))


#For Git/Server deployment

# # Initialize API clients
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# mistral_client = MistralClient(api_key=st.secrets["MISTRAL_API_KEY"])

# def check_password():
#     """Returns `True` if the user has entered the correct password."""

#     def password_entered():
#         """Checks whether the password entered is correct."""
#         if st.session_state["password"] == st.secrets["ACCESS_PASSWORD"]:
#             st.session_state["password_correct"] = True

#FOR LOCAL MACHINE

def check_password():
    """Returns `True` if the user has entered the correct password."""

    def password_entered():
        """Checks whether the password entered is correct."""
        if st.session_state["password"] == os.getenv("ACCESS_PASSWORD"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.markdown("""
            <style>
            .auth-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 2rem;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("## 🔐 Access Code Required - LLM 🧠 Comparison Tool")
            st.text_input("Enter Access Code 🔑:", type="password", key="password")
            st.button("Submit", on_click=password_entered, type="primary")
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("## 🔐 Access Code Required - LLM 🧠 Comparison Tool")
            st.text_input("Enter Access Code 🔑:", type="password", key="password")
            st.button("Submit", on_click=password_entered, type="primary")
            st.error("😕 Ooops! Incorrect Code. Get in touch with Basant.")
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    else:
        # Password correct
        return True

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = {
            "gemini": [],
            "openai": [],
            "mistral": [],
            "deepseek": [],
            "llama": [],
            "qwen": []
        }
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1000
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = ["gemini", "deepseek", "qwen"]  # default selection

def check_creator_question(question):
    """Check if the question is asking about the creator/developer"""
    creator_keywords = [
        "who created", "who made", "who developed", "who built",
        "who is the creator", "who is the developer", "who's the creator",
        "who designed", "creator of", "developer of", "built by"
    ]
    return any(keyword in question.lower() for keyword in creator_keywords)

def get_gemini_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This Comparison Tool is created by Basant Singh a Product Manager at Whizlabs."
    
    model = genai.GenerativeModel('gemini-pro')
    #model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(
        messages[-1]["content"] if messages else "",
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
    )
    return response.text

def get_openai_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This LLM Comparison Tool is created by Basant Singh - Product Manager, Whizlabs."
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def get_mistral_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This LLM Comparison Tool is developed by Basant Singh a Product Manager at Whizlabs."
    
    response = mistral_client.chat(
        model="mistral-tiny",
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def get_deepseek_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This LLM Comparison Tool is created by Basant Singh - Product Manager, Whizlabs."
    
    response = fireworks_client.chat.completions.create(
        model="accounts/fireworks/models/deepseek-v3",
        #models="accounts/fireworks/models/deepseek-r1",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def get_llama_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This LLM Comparison Tool is created by Basant Singh - Product Manager, Whizlabs."
    
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def get_qwen_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This LLM Comparison Tool is created by Basant Singh - Product Manager, Whizlabs."
    
    response = together_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# Add this before the process_user_input function
response_functions = {
    "gemini": get_gemini_response,
    "openai": get_openai_response,
    "mistral": get_mistral_response,
    "deepseek": get_deepseek_response,
    "llama": get_llama_response,
    "qwen": get_qwen_response
}

def handle_send():
    """Callback function to handle send button click"""
    if st.session_state.user_input:
        prompt = st.session_state.user_input
        
        # Add user message to each model's chat history
        for model in st.session_state.selected_models:
            st.session_state.messages[model].append({"role": "user", "content": prompt})

        # Get responses from selected models
        for model in st.session_state.selected_models:
            try:
                response = response_functions[model](
                    st.session_state.messages[model],
                    st.session_state.temperature,
                    st.session_state.max_tokens
                )
                st.session_state.messages[model].append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Error from {model}: {str(e)}"
                st.error(error_message)
                st.session_state.messages[model].append({"role": "assistant", "content": error_message})
        
        # Clear input after processing
        st.session_state.user_input = ""

def clear_chat():
    """Clear all chat messages for all models"""
    # Only clear the messages
    for model in st.session_state.messages:
        st.session_state.messages[model] = []
    # Force a rerun to reset the UI
    st.rerun()

def handle_clear():
    """Callback function for clear button"""
    # Clear messages
    for model in st.session_state.messages:
        st.session_state.messages[model] = []
    # Reset the input by setting it to empty string
    if "user_input" in st.session_state:
        st.session_state.user_input = ""

def validate_api_keys():
    required_keys = {
        "GOOGLE_API_KEY": "Google Gemini",
        "OPENAI_API_KEY": "OpenAI",
        "MISTRAL_API_KEY": "Mistral",
        "FIREWORKS_DS_API_KEY": "DeepSeek",
        "TOGETHER_ML_API_KEY": "Together/Llama"
    }
    
    missing_keys = []
    for key, service in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(service)
    
    if missing_keys:
        st.error(f"Missing API keys for: {', '.join(missing_keys)}")
        return False
    return True

def main():
    if not validate_api_keys():
        return
    st.set_page_config(
        page_title="LLM Comparison Tool",
        page_icon="🤖",
        layout="wide"
    )
    
    if check_password():
        initialize_session_state()
        
        # Update the custom CSS section with text area styling
        st.markdown("""
            <style>
            .stApp {
                background-color: #f5f5f5;
            }
            .chat-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .user-message {
                background-color: #e3f2fd;
            }
            .assistant-message {
                background-color: white;
            }
            .model-header {
                font-size: 1.5rem;
                font-weight: bold;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .vertical-divider {
                border-left: 3px solid #FFD700;
                height: 50px;
                margin: 0 auto;
                margin-top: 10px;
            }
            
            /* Enhanced text area styling */
            .stTextArea textarea {
                background-color: white !important;
                border: 2px solid #e0e0e0 !important;
                border-radius: 10px !important;
                padding: 10px !important;
                font-size: 16px !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
                color: #333 !important;
            }
            
            .stTextArea textarea:focus {
                border-color: #007bff !important;
                box-shadow: 0 0 0 2px rgba(0,123,255,0.25) !important;
            }

            /* Style for placeholder text */
            .stTextArea textarea::placeholder {
                color: #666 !important;
                opacity: 1 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("# LLM Comparison Tool - ChatB🤖ttle Royale👑")
        #st.markdown("### 🔄 A tool by Basant to compare responses from different AI models in real-time!")
        st.markdown('<p style="color: gray; font-size: 20px;">A tool by Basant to compare responses from different AI models in real-time!</p>', unsafe_allow_html=True)

        # Settings in sidebar
        with st.sidebar:
            st.markdown("### 🤖 Select Models")
            
            # Define available models
            available_models = {
                "gemini": "🧠 Google Gemini",
                "openai": "🎯 OpenAI GPT-3.5",
                "mistral": "⚡ Mistral AI",
                "deepseek": "🔍 DeepSeek V3",
                "llama": "🦙 Meta Llama 3",
                "qwen": "🌟 Qwen 2.5"
            }
            
            # Model selection
            selected = []
            for model, label in available_models.items():
                if st.checkbox(label, value=model in st.session_state.selected_models):
                    selected.append(model)
            
            # Ensure exactly 3 models are selected with red warning text
            if len(selected) != 3:
                st.markdown(
                    '<p style="color: #FF0000; font-weight: bold; font-size: 16px;">⚠️ Please select exactly 3 models for comparison</p>', 
                    unsafe_allow_html=True
                )
            else:
                st.session_state.selected_models = selected

            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            
            # Temperature slider with emoji
            st.markdown("#### 🌡️ Creativity / Temperature")
            st.session_state.temperature = st.slider(
                "Control creativity / randomness",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher value results in higher level of creativity."
            )
            
            # Max tokens slider with emoji
            st.markdown("#### 📝 Max Tokens (~ Words)")
            st.session_state.max_tokens = st.slider(
                "Control response length",
                min_value=100,
                max_value=1000,
                value=300,
                step=100,
                help="Maximum number of tokens in the response"
            )
            
            # Add a divider
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.info("Try different temperatures and token limits to see how they affect the responses!")

        # Only proceed with display if exactly 3 models are selected
        if len(st.session_state.selected_models) == 3:
            # Create columns with dividers for selected models
            cols = st.columns([0.3, 0.01, 0.3, 0.01, 0.3])
            
            # Display messages for selected models
            for i, model in enumerate(st.session_state.selected_models):
                col_index = i * 2  # Skip divider columns
                with cols[col_index]:
                    st.markdown(f'<div class="model-header">{available_models[model]}</div>', unsafe_allow_html=True)
                    for message in st.session_state.messages[model]:
                        div_class = "user-message" if message["role"] == "user" else "assistant-message"
                        emoji = "👤" if message["role"] == "user" else "🤖"
                        st.markdown(f"""
                            <div class="chat-message {div_class}">
                                <b>{emoji} {message["role"].title()}:</b> {message["content"]}
                            </div>
                        """, unsafe_allow_html=True)
                
                # Add divider if not the last model
                if i < 2:
                    with cols[col_index + 1]:
                        st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

        # Input and button section
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.text_area(
                "Your Message to LLMs",
                key="user_input",
                height=100,
                placeholder="Type your message here...",
                label_visibility="collapsed"
            )
        
        with col2:
            st.button(
                "Send 🚀", 
                on_click=handle_send,
                use_container_width=True
            )

        # Clear chat button with callback
        st.button(
            "🗑️ Clear Chat", 
            on_click=handle_clear,
            use_container_width=True, 
            key="clear_button"
        )

if __name__ == "__main__":
    main() 
