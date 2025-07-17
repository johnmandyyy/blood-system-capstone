from django.shortcuts import redirect
from django.http import HttpRequest
import app.constants.template_constants as Templates
from django.contrib.auth import logout, authenticate, login

class TemplateView:
    """Built in Template Renderer View Level"""

    def __init__(self):
        pass

    def index(self, request):
        """Renders the index page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        return Templates.INDEX.render_page(request)

    def reports(self, request):
        """Renders the reports page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        return Templates.REPORTS.render_page(request)

    def datasets(self, request):
        """Renders the datasets page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        return Templates.DATASETS.render_page(request)

    def maintenance(self, request):
        """Renders the maintenance page."""

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        return Templates.MAINTENANCE.render_page(request)

    def login(self, request):
        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return Templates.LOGIN.render_page(request)

        return redirect("home")
            
    def user_logout(self, request):
        logout(request)
        return redirect("login")

    