from django.shortcuts import redirect
from django.http import HttpRequest, HttpResponse
import app.constants.template_constants as Templates
from django.contrib.auth import logout, authenticate, login
from app.helpers import authentication
from app.defined_api.cnn_api import CNN
import pdfkit
from app.defined_api.emailing import Emailing
from app.data.prescriptions import Prescription
from app.defined_api.check_internet import Pinger
# Data Access Layer
from app.data.patient import PatientRecord

class TemplateView:
    """Built in Template Renderer View Level"""

    def __init__(self):
        pass

    def profile(self, request):
        """Renders the profile page """

        assert isinstance(request, HttpRequest)
        if not request.user.is_authenticated:
            return redirect("login")
        
        return Templates.PROFILE.render_page(request)

    def prescription(self, request, id):
        """ Renders the prescription page. """
        
        P = Prescription(id)
        records = P._get_prescription_record()
        
        return Templates.PRESCRIPTION.addContext(records).render_page(request)

    def download_prescription(self, request, id):
        """ A method to download prescription. """

        url = "http://localhost:8000/prescription/" + str(id) + "/"

        # Path to wkhtmltopdf
        path_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

        P = Prescription(id)
        records = P._get_prescription_record()
        email = records["email"]
        patient_name = records["patient_name"]
        pathologist_name = records["attending_physician"]
        # Generate PDF from the URL
        pdf_bytes = None

        try:
            # Generate PDF as bytes (not file on disk)
            filename = f"prescription_{id}.pdf"
            pdf_bytes = pdfkit.from_url(url, filename, configuration=config)
            

            # Check Internet First
            e = Emailing()

            message = f"""<h1>Hello, {patient_name} I hope this message finds you well.</h1>
            <p>Attached is <b>the prescription</b>.</p><br><br>Regards,<br>{pathologist_name}"""

            e.send_email(
                to=[email],
                subject="Prescription and Findings (Lymphoblastic / CNN)",
                body="This is a plain text email.",
                html=message,
                attachments=[filename]
            )
            
            # Send as downloadable file
            response = HttpResponse(pdf_bytes, content_type="application/pdf")
            response["Content-Disposition"] = f'attachment; filename="prescription_{id}.pdf"'
            return response

        except Exception as e:
            print(e, 'is the exception.', type(e))
            return redirect("patients_records")
        
        
    def home(self, request):
        """Renders the home page."""
       
        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        return Templates.HOME.render_page(request)
    
    def temporary_method(self):
        from app.models import Images
        from app import settings
        import os
        import shutil
        print(os.getcwd())

        validation_sets = Images.objects.filter(used_for='Validation')

        # Define the destination folder
        destination_folder = os.path.join(settings.MEDIA_ROOT, 'test_folder')

        # Ensure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)

        for picture in validation_sets:
            source_path = os.path.join(settings.MEDIA_ROOT, str(picture.location))  # Assuming `image` is the field storing the file path
            destination_path = os.path.join(destination_folder, os.path.basename(source_path))

            if os.path.exists(source_path):  # Ensure the file exists before copying
                shutil.copy(source_path, destination_path)
                print(f"Copied {source_path} to {destination_path}")
            else:
                print(f"File not found: {source_path}")
          
    def pathologist(self, request):
        """Renders the datasets page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        if request.user.is_superuser == False:
            return Templates.PATHOLOGIST.render_page(request)
        
        return redirect("home")
    
    def datasets(self, request):
        """Renders the datasets page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        if request.user.is_superuser == True:
            return Templates.DATASETS.render_page(request)
        
        return redirect("login")

    def patients_records(self, request, id):
        """Renders the patients record page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")
        
        patient = PatientRecord()
        if request.user.is_superuser == False and id is not None and patient.get_patient(id) is not None:
            return Templates.PATIENTS_RECORD.render_page(request)

        return redirect("home")
    
    def patients(self, request):
        """Renders the patients page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        if request.user.is_superuser == False:
            return Templates.PATIENTS.render_page(request)

        return redirect("home")

    def administration(self, request):
        """Renders the administration page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")
        
        if request.user.is_superuser == True:
            return Templates.ADMINISTRATION.render_page(request)
        
        return redirect("login")

    def login(self, request):
        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return Templates.LOGIN.render_page(request)
        return redirect("home")  # Change the home to your index page.

    def user_logout(self, request):
        logout(request)
        return redirect("login")
