# import streamlit as st
# from backend import process_emails

# st.set_page_config(page_title="Email Sorting Assistant", layout="wide")
# st.title("📬 Email Sorting Assistant")

# st.markdown("Paste your emails below (one per line):")

# email_input = st.text_area("Emails", height=300)

# if st.button("Categorize Emails"):
#     if email_input.strip():
#         email_list = [e.strip() for e in email_input.split("\n") if e.strip()]
#         results = process_emails(email_list)

#         st.subheader("📊 Categorized Emails")
#         for item in results:
#             st.markdown(f"**Priority:** `{item['priority']}`")
#             st.markdown(f"> {item['email']}")
#             st.markdown("---")
#     else:
#         st.warning("Please enter at least one email.")


import streamlit as st
from backend import process_emails

st.set_page_config(page_title="Email Sorting Assistant", layout="centered")
st.title("📬 Email Sorting Assistant")

st.markdown("Enter a single email below to categorize its priority:")

# Single email input
email_input = st.text_area("Email Content", height=200)

if st.button("Categorize Email"):
    if email_input.strip():
        # Wrap the single email in a list for backend compatibility
        results = process_emails([email_input.strip()])

        # Display result
        priority = results[0]['priority']
        st.success(f"**Priority:** `{priority}`")
        st.markdown(f"> {email_input.strip()}")
    else:
        st.warning("Please enter an email to categorize.")
