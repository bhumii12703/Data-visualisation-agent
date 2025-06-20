import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from together import Together
import plotly.express as px
import plotly.graph_objects as go

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

def get_llm_response(messages):
    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        try:
            response = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=messages,
                stream=False
            )

            if response and hasattr(response, 'message'):
                message_content = response.message.content
                python_code = match_code_blocks(message_content)
                return python_code, message_content
            else:
                st.error("Failed to get response from LLM")
                return None, "Error: No response from LLM"

        except Exception as e:
            st.error(f"Error calling LLM: {str(e)}")
            return None, f"Error: {str(e)}"

def match_code_blocks(text):
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def execute_visualization(code, df):
    try:
        # Create a local namespace for execution
        local_namespace = {'df': df, 'plt': plt, 'sns': sns, 'px': px, 'go': go}
        
        # Execute the code within this namespace
        exec(code, globals(), local_namespace)
        
        # If using matplotlib/seaborn, get the current figure
        if 'plt' in code:
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close()
        
        return True
    except Exception as e:
        st.error(f"Error executing visualization code: {str(e)}")
        return False

def main():
    st.title("AI Data Visualization Agent")
    initialize_session_state()

    # API Key input
    api_key = st.text_input("Enter your Together AI API Key:", 
                           value=st.session_state.together_api_key, 
                           type="password")
    if api_key:
        st.session_state.together_api_key = api_key

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        # User input
        user_input = st.text_area("Describe the visualization you want:", 
                                 height=100)
        
        if st.button("Generate Visualization"):
            if not st.session_state.together_api_key:
                st.error("Please enter your Together AI API key first.")
                return

            # Prepare the message
            messages = [
                {"role": "system", "content": "You are a data visualization expert. Generate Python code using pandas, matplotlib, seaborn, or plotly to create visualizations. Only respond with Python code within ```python``` blocks."},
                {"role": "user", "content": f"Create a visualization for this data: {user_input}. Here are the columns available: {', '.join(df.columns)}"}
            ]

            # Get LLM response
            code, full_response = get_llm_response(messages)
            
            if code:
                st.write("Generated Code:")
                st.code(code, language="python")
                
                st.write("Visualization:")
                execute_visualization(code, df)
            else:
                st.error("Failed to generate visualization code.")

if __name__ == "__main__":
    main()