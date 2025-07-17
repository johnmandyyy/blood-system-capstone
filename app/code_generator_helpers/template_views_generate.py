import sys
import os
def generate_template_class(object_name, methods, class_desc="A template level views."):
    class_template = """from django.shortcuts import redirect
from django.http import HttpRequest
import app.constants.template_constants as Templates
from django.contrib.auth import logout, authenticate, login

class TemplateView:
    \"\"\"Built in Template Renderer View Level\"\"\"

    def __init__(self):
        pass
"""
    method_template = """
    def {method_name}(self, request):
        \"\"\"Renders the {method_desc} page.\"\"\"

        assert isinstance(request, HttpRequest)

        if not request.user.is_authenticated:
            return redirect("login")

        return Templates.{template_name}.render_page(request)
"""

    methods_code = ""
    for method in methods:
        method_name, method_desc = (
            method.split(":") if ":" in method else (method, method)
        )
        method_name = method_name.lower()
        method_desc = method_desc.lower()
        template_name = method_name.upper()
        methods_code += method_template.format(
            method_name=method_name,
            method_desc=method_desc,
            template_name=template_name,
        )

    full_class_code = class_template + methods_code
    full_class_code = full_class_code + """
    def login(self, request):
        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return Templates.LOGIN.render_page(request)

        return redirect("home") #Change the home to your index page.

    def authenticate_user(self, request):
        try:
            if request.method == "POST":

                username = request.POST.get("username")
                password = request.POST.get("password")
                user = authenticate(request, username=username, password=password)
                if user is not None:
                    login(request, user)  # Library level not instance.
                    return redirect("index")  # Change the home to your index page.

        except Exception as e:
            pass

        return redirect("login")

    def user_logout(self, request):
        logout(request)
        return redirect("login")

    """
    return full_class_code


def generate_builder_code(object_name, methods):
    builder_template = """from app.builder.template_builder import Builder
from app.constants import app_constants
from app.builder.forms_builder import FormBuilder
"""
    builder_code = ""
    for method in methods:
        method_name, method_desc = (
            method.split(":") if ":" in method else (method, method)
        )
        template_name = method_name.upper()
        page_path = f"app/{method_name}.html"
        title = f"{method_desc.lower()}"

        builder_code += f"""
{template_name} = (
    Builder()
    .addPage("{page_path}")
    .addTitle("{title}")
)

{template_name}.build()
"""

    # Append LOGIN code
    builder_code += """
LOGIN = (
    Builder()
    .addPage("app/login.html")
    .addTitle("login")
    .addContext(
        {
            "title": "Login - Page",
            "obj_name": "login",
            "app_name": app_constants.SOFTWARE_NAME,
            "app_desc": app_constants.SOFTWARE_DESCRIPTION
        }
    )
)
LOGIN.build()
"""

    full_builder_code = builder_template + builder_code
    return full_builder_code


def generate_html_vue_code(filename):
    return """{% extends "app/layout.html" %}
{% load static %}
{% load crispy_forms_tags %}
{% block content %}

<div id="{{ obj_name }}">
    <div class="row">

    </div>
</div>

<script>

new Vue({
    delimiters: ["[[", "]]"],
    el: "#" + '{{ obj_name }}',
    data: {},
    mounted() {
        if (document.querySelector("#" + '{{ obj_name }}')) {
            console.log("Mounted " + '{{ obj_name }}' + " page.")
        }
    },
    methods: {},
});
</script>
{% endblock %}"""


def generate_urls_code(object_name, methods):
    url_template = f"""\"\"\"
Definition of urls for app.
\"\"\"

from django.urls import path, include
from django.contrib import admin
from app.views import TemplateView
from app.api import *
import app.constants.url_constants as URLConstants
from app.constants import app_constants

MainView = TemplateView()

list_create_patterns = URLConstants.GenericAPI.list_create_patterns
get_update_destroy_patterns = URLConstants.GenericAPI.retrieve_update_delete_patterns

api_patterns = [
    path("api/", include((list_create_patterns, app_constants.APP_NAME))),
    path("api/", include((get_update_destroy_patterns, app_constants.APP_NAME))),
]

template_patterns = [
"""

    url_patterns_code = ""
    for method in methods:
        method_name = method.split(":")[0] if ":" in method else method
        url_patterns_code += f'    path("{method_name}/", MainView.{method_name}, name="{method_name}"),\n'

    url_patterns_code += """    path("admin/", admin.site.urls),\n    path("logout/", MainView.user_logout, name="logout"),\n    path("login/", MainView.login, name="login"),\n    path("authenticate_user/", MainView.authenticate_user, name="authenticate_user")
]

urlpatterns = template_patterns + api_patterns
"""

    full_url_code = url_template + url_patterns_code
    return full_url_code


def generate_partials(filename, methods):
    partial_html = """<div class="offcanvas offcanvas-start" tabindex="-1" id="offcanvasNavbar" aria-labelledby="offcanvasNavbarLabel">
    <div class="offcanvas-header">
        <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></button>
    </div>
    <div class="offcanvas-body">

        <div class="text-center">
            <h1 class="display-3"><i class="fa-solid fa-circle-user"></i></h1>
            <p class="lead">Welcome, {{ request.user }}!</p>
        </div>

        <ul class="navbar-nav">
    """

    for method in methods:
        partial_html += f"""
        <li class="nav-item">
            <a class="nav-link fw-normal text-muted" href="{{% url '{method}' %}}">
                <i class="fa-solid fa-icons p-3"></i>{method.capitalize()}
            </a>
        </li>
        """

    partial_html += """
        <li class="nav-item">
            <a class="nav-link fw-normal text-muted" href="{% url 'logout' %}">
                <i class="fa-solid fa-right-from-bracket p-3"></i>Sign-out
            </a>
        </li>
        </ul>
    </div>
    </div>
    """
    return partial_html




def write_to_file(filename, content, path):
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Write content to the file
    with open(os.path.join(path, filename), "w") as file:
        file.write(content)

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print(
            "Usage: python generate_template.py <method1[:desc1]> <method2[:desc2]> ... <methodN[:descN]> [--class-desc <description>]"
        )
        sys.exit(1)

    class_desc = "A template level views."
    if "--class-desc" in sys.argv:
        class_desc_index = sys.argv.index("--class-desc")
        class_desc = sys.argv[class_desc_index + 1]
        methods = sys.argv[2:class_desc_index]
    else:
        methods = sys.argv[2:]

    #object_name = sys.argv[1]

    object_name = ""

    generated_class_code = generate_template_class(object_name, methods, class_desc)
    write_to_file("views.py", generated_class_code, "app")

    generated_builder_code = generate_builder_code(object_name, methods)
    write_to_file("template_constants.py", generated_builder_code, "app/constants")

    for each_rows in methods:
        generated_html_code = generate_html_vue_code(str(each_rows).lower())
        write_to_file(str(each_rows).lower() + ".html", generated_html_code, "app/templates/app")

    print(f"Generated HTML Vue Templates")

    generated_urls_code = generate_urls_code(object_name, methods)
    write_to_file("urls.py", generated_urls_code, "app")

    generated_partials = generate_partials(object_name, methods)
    write_to_file("sidebar.html", generated_partials, "app/templates/app/partials")

    print(f"Generated views.py with class view.")
    print(f"Generated builders.py with template_constants setup.")
