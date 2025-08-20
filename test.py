import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None
    
import base64
from email.mime.text import MIMEText

def get_last_n_emails(service, n=10):
    try:
        # Get messages from the inbox
        results = service.users().messages().list(userId='me', maxResults=n).execute()
        messages = results.get('messages', [])

        if not messages:
            print('No messages found.')
            return

        print(f"Listing subjects of the last {n} emails:")
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
            
            # Extract email subject
            for header in msg['payload']['headers']:
                if header['name'] == 'Subject':
                    print(f"- {header['value']}")
                    break

    except HttpError as error:
        print(f'An error occurred: {error}')

if __name__ == '__main__':
    service = get_gmail_service()
    if service:
        get_last_n_emails(service, n=10)