import smtplib
import mimetypes
import os
from email.message import EmailMessage
from app.constants import app_constants
from rest_framework.views import APIView
from rest_framework.response import Response

class Emailing(object):
    
    def __init__(self):
        """ A constructor for sending email. """
  
        pass


    def send_email(self, to, subject, body=None, html=None, attachments=None, cc=None, bcc=None):
        """
        Send an email via Gmail SMTP using only built-in Python libraries.

        Args:
            send_email(
                to=["sample@mail.com"],
                subject="Test Email",
                body="This is a plain text email.",
                html="<h1>Hello!</h1><p>This is <b>HTML</b> content.</p>",
                attachments=[]
            )
        """
        try:

            sender = app_constants.SENDER
            app_password = app_constants.PASSWORD

            msg = EmailMessage()
            msg["From"] = sender
            msg["To"] = ", ".join(to)

            if cc:
                msg["Cc"] = ", ".join(cc)
            msg["Subject"] = subject

            # Add body (plain + HTML alternative)
            if body and html:
                msg.set_content(body)
                msg.add_alternative(html, subtype="html")
            elif html:
                msg.set_content("Your client does not support HTML.")
                msg.add_alternative(html, subtype="html")
            else:
                msg.set_content(body or "")

            # Attach files
            for path in attachments or []:
                with open(path, "rb") as f:
                    data = f.read()
                mtype, _ = mimetypes.guess_type(path)
                maintype, subtype = mtype.split("/", 1) if mtype else ("application", "octet-stream")
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(path))

            # Final recipient list (To + CC + BCC)
            recipients = list(set((to or []) + (cc or []) + (bcc or [])))

            # Connect & send
            with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
                smtp.starttls()
                smtp.login(sender, app_password)
                smtp.send_message(msg, from_addr=sender, to_addrs=recipients)

            print(f"✅ Email sent to {', '.join(recipients)}")

        except Exception as e:
            pass



class EmailingAPI(APIView):

    def __init__(self):
        pass

    def post(self, request):
        
        try:
            payload = {
                'to': [request.data['to']],
                'subject': request.data['subject'],
                'body': request.data['body'],
                'html': request.data['html'],
                'cc': request.data['cc'],
                'bcc': request.data['bcc']
            }

            E = Emailing()
            E.send_email(
                to = payload['to'],
                subject = payload['subject'],
                body = payload['body'],
                html = payload['html'],
                cc = payload['cc'],
                bcc = payload['bcc']
            )

            return Response({"message": "Email has been sent."}, 200)
        
        except Exception as e:

            return Response({"message": "Email wasn't sent."}, 400)




