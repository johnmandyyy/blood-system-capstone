from django.shortcuts import render
from datetime import datetime
from app.logs.logging import Logger
from app.constants import app_constants
import inspect

class TemplateBuilder:
    """A template builder class."""

    def __init__(self):

        self.__Page = None
        self.__Title = None

    # --------------------------------------------------------------------------------------------------------------- #
    def setPage(self, page: str) -> None:
        """A method used to set page location"""
        self.__Page = page

    # --------------------------------------------------------------------------------------------------------------- #
    def setTitle(self, title: str) -> None:
        """A method used to set title"""
        self.__Title = title

    # --------------------------------------------------------------------------------------------------------------- #
    def getProps(self):
        """Getters"""
        return {"page": self.__Page, "title": self.__Title}

class Builder:

    def __init__(self):
        """A builder template for views in HTML."""

        self.instance = TemplateBuilder()

        self.Page = None
        self.Context = None
        self.Object = None

        
        self.initialize()

    # --------------------------------------------------------------------------------------------------------------- #
    def addPage(self, page) -> TemplateBuilder:
        """To add page."""

        self.instance.setPage(page)
        return self

    # --------------------------------------------------------------------------------------------------------------- #
    def addTitle(self, title) -> TemplateBuilder:
        """To add title to a page."""
        self.instance.setTitle(title)
        return self

    # --------------------------------------------------------------------------------------------------------------- #
    def addContext(self, context={}) -> TemplateBuilder:
        """Use to override server side variables in views."""
    
        if len(context) < 1:
            context = self.Context
        else:
            self.Context = context

        self.Context = context
        return self

    # --------------------------------------------------------------------------------------------------------------- #
    def initialize(self):
        """Used to initialize context."""
        
        self.Context = {
            "runtime_instances": None if False else self.Object,
            "title": 'Generic Page',
            "date": str(datetime.now()),
            "obj_name": str(self.instance.getProps()["title"]).lower(),
        }


    # --------------------------------------------------------------------------------------------------------------- #
    def build(self) -> TemplateBuilder:
        """A method used to build the object."""

        self.Page = self.instance.getProps()["page"]
        return self.instance

    # --------------------------------------------------------------------------------------------------------------- #
    def render_page(self, request):
        """A method to render when there is an error in the page."""

        try:
            
            page = render(request, self.Page, self.Context)
            
            Logger(
                message="Loaded Page",
                source=__name__,
                request=request,
                level=app_constants.LOG_LEVEL.INFO,
                log_type=app_constants.LOG_TYPE.HTTP_REQUEST,
                response_status= page.status_code,
            )

            return page

        except Exception as e:

            page = render(
                request,
                "app/constants/error.html",
                {
                    "title": "Maintenance Page",
                    "date": str(datetime.now()),
                    "message": str(e),
                },
            )

            Logger(
                message="Loaded Page Handler Page",
                source=__name__,
                request=request,
                level=app_constants.LOG_LEVEL.ERROR,
                log_type=app_constants.LOG_TYPE.HTTP_REQUEST,
                exception=e,
                response_status= page.status_code
            )

            return page

