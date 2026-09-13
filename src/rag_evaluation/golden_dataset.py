"""Golden Q&A dataset for the FastAPI documentation corpus.

The dataset is defined here in code (``QUESTIONS``) so it can be loaded with
``load_golden_dataset()`` without touching disk, and is also exported to
``data/golden_dataset.json`` in the legacy schema for the class-based loader.

Every lookup and multi_hop ``golden_answer`` is traceable to verbatim text in
``data/raw/fastapi_docs/**/*.md``. no_answer questions are deliberately
plausible-but-absent: the FastAPI docs do not cover those topics.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

CATEGORIES = ("lookup", "multi_hop", "no_answer", "ambiguous")


@dataclass
class GoldenQuestion:
    id: str
    category: str
    question: str
    answer: str
    source_documents: List[str] = field(default_factory=list)
    source_sections: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_lookup(self) -> bool:
        return self.category == "lookup"

    @property
    def is_multi_hop(self) -> bool:
        return self.category == "multi_hop"

    @property
    def is_no_answer(self) -> bool:
        return self.category == "no_answer"

    @property
    def is_ambiguous(self) -> bool:
        return self.category == "ambiguous"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "source_documents": self.source_documents,
            "source_sections": self.source_sections,
            "notes": self.notes,
        }


QUESTIONS = [
    # ------------------------------------------------------------------
    # lookup (20) — single fact, single document, verbatim from the docs
    # ------------------------------------------------------------------
    {
        "id": "q001",
        "question": "What is FastAPI?",
        "golden_answer": "FastAPI is a modern, fast (high-performance), web framework for "
            "building APIs with Python based on standard Python type hints.",
        "type": "lookup",
        "source_docs": ["index.md"],
        "section": "FastAPI",
        "notes": "Verbatim from the index page opening definition.",
    },
    {
        "id": "q002",
        "question": "In the FastAPI docs query-parameters example, what are the default values of skip and limit?",
        "golden_answer": "In the example above they have default values of `skip=0` and `limit=10`.",
        "type": "lookup",
        "source_docs": ["tutorial/query-params.md"],
        "section": "Defaults",
        "notes": "Verbatim single-fact answer from the Defaults section.",
    },
    {
        "id": "q003",
        "question": "When item_id is declared as an int, what JSON does the path-parameters example return for /items/3?",
        "golden_answer": '{"item_id": 3}',
        "type": "lookup",
        "source_docs": ["tutorial/path-params.md"],
        "section": "Data conversion",
        "notes": "Verbatim response; the doc notes the value is 3 as a Python int, not the string \"3\".",
    },
    {
        "id": "q004",
        "question": "What response body does the simplest FastAPI app with @app.get(\"/\") return?",
        "golden_answer": '{"message": "Hello World"}',
        "type": "lookup",
        "source_docs": ["tutorial/first-steps.md"],
        "section": "Check it",
        "notes": "Verbatim JSON returned by the first-steps example.",
    },
    {
        "id": "q005",
        "question": "Which HTTP methods does the FastAPI docs describe as the ones normally used to create, read, update, and delete data?",
        "golden_answer": "POST: to create data. GET: to read data. PUT: to update data. DELETE: to delete data.",
        "type": "lookup",
        "source_docs": ["tutorial/first-steps.md"],
        "section": "Operation",
        "notes": "Verbatim mapping of methods to actions from the first-steps recap.",
    },
    {
        "id": "q006",
        "question": "What do you import to test a FastAPI application, and where does it come from?",
        "golden_answer": "from fastapi.testclient import TestClient — FastAPI provides the same starlette.testclient as fastapi.testclient just as a convenience, but it comes directly from Starlette.",
        "type": "lookup",
        "source_docs": ["tutorial/testing.md"],
        "section": "Using TestClient",
        "notes": "Verbatim: FastAPI mirrors starlette.testclient; TestClient is used like httpx.",
    },
    {
        "id": "q007",
        "question": "According to the FastAPI docs, what is a middleware?",
        "golden_answer": "A \"middleware\" is a function that works with every request before it is processed by any specific path operation. And also with every response before returning it.",
        "type": "lookup",
        "source_docs": ["tutorial/middleware.md"],
        "section": "Middleware",
        "notes": "Verbatim definition from the middleware tutorial.",
    },
    {
        "id": "q008",
        "question": "When do FastAPI background tasks run relative to the response?",
        "golden_answer": "You can define background tasks to be run after returning a response.",
        "type": "lookup",
        "source_docs": ["tutorial/background-tasks.md"],
        "section": "Background Tasks",
        "notes": "Verbatim opening statement of the background-tasks tutorial.",
    },
    {
        "id": "q009",
        "question": "Which FastAPI extra data type is represented as a float of total seconds in requests and responses?",
        "golden_answer": "datetime.timedelta: In requests and responses will be represented as a float of total seconds.",
        "type": "lookup",
        "source_docs": ["tutorial/extra-data-types.md"],
        "section": "Other data types",
        "notes": "Verbatim from the extra-data-types list.",
    },
    {
        "id": "q010",
        "question": "In what format is a datetime.datetime represented in FastAPI requests and responses?",
        "golden_answer": "In requests and responses will be represented as a str in ISO 8601 format, like: 2008-09-15T15:53:00+05:00.",
        "type": "lookup",
        "source_docs": ["tutorial/extra-data-types.md"],
        "section": "Other data types",
        "notes": "Verbatim from the extra-data-types list.",
    },
    {
        "id": "q011",
        "question": "Which string validation parameters can be added to a Query parameter according to the docs?",
        "golden_answer": "Validations specific for strings: min_length, max_length, pattern.",
        "type": "lookup",
        "source_docs": ["tutorial/query-params-str-validations.md"],
        "section": "Recap",
        "notes": "Verbatim recap of string-specific validations.",
    },
    {
        "id": "q012",
        "question": "What are the four numeric validation parameters FastAPI provides for Query and Path?",
        "golden_answer": "gt: greater than; ge: greater than or equal; lt: less than; le: less than or equal.",
        "type": "lookup",
        "source_docs": ["tutorial/path-params-numeric-validations.md"],
        "section": "Recap",
        "notes": "Verbatim numeric-validation recap.",
    },
    {
        "id": "q013",
        "question": "What is the default HTTP status code for a FastAPI path operation response?",
        "golden_answer": "200 is the default status code, which means everything was \"OK\".",
        "type": "lookup",
        "source_docs": ["tutorial/response-status-code.md"],
        "section": "About HTTP status codes",
        "notes": "Verbatim from the HTTP status-code overview.",
    },
    {
        "id": "q014",
        "question": "According to the FastAPI CORS docs, what makes up an origin?",
        "golden_answer": "An origin is the combination of protocol (http, https), domain (myapp.com, localhost, localhost.tiangolo.com), and port (80, 443, 8080).",
        "type": "lookup",
        "source_docs": ["tutorial/cors.md"],
        "section": "Origin",
        "notes": "Verbatim definition of an origin.",
    },
    {
        "id": "q015",
        "question": "How do you return an HTTP error response to a client in FastAPI?",
        "golden_answer": "To return HTTP responses with errors to the client you use HTTPException.",
        "type": "lookup",
        "source_docs": ["tutorial/handling-errors.md"],
        "section": "Use HTTPException",
        "notes": "Verbatim from the handling-errors tutorial.",
    },
    {
        "id": "q016",
        "question": "Which package must be installed before FastAPI can receive uploaded files?",
        "golden_answer": "python-multipart.",
        "type": "lookup",
        "source_docs": ["tutorial/request-files.md"],
        "section": "Request Files",
        "notes": "Verbatim: uploaded files are sent as form data, which requires python-multipart.",
    },
    {
        "id": "q017",
        "question": "By default, how does the Header parameter convert parameter names?",
        "golden_answer": "By default, Header will convert the parameter names characters from underscore (_) to hyphen (-) to extract and document the headers.",
        "type": "lookup",
        "source_docs": ["tutorial/header-params.md"],
        "section": "Automatic conversion",
        "notes": "Verbatim from the automatic-conversion section.",
    },
    {
        "id": "q018",
        "question": "What is an environment variable according to the FastAPI docs?",
        "golden_answer": "An environment variable (also known as an env var) is a value that lives outside of your Python code, in the operating system, and can be read by your application and other programs.",
        "type": "lookup",
        "source_docs": ["environment-variables.md"],
        "section": "Environment Variables",
        "notes": "Verbatim opening definition.",
    },
    {
        "id": "q019",
        "question": "In the command uvicorn main:app, what do main and app refer to?",
        "golden_answer": "main: the file main.py (the Python \"module\"). app: the object created inside of main.py with the line app = FastAPI().",
        "type": "lookup",
        "source_docs": ["deployment/manually.md"],
        "section": "Run a Server Manually",
        "notes": "Verbatim explanation of the uvicorn import string.",
    },
    {
        "id": "q020",
        "question": "Which command serves a FastAPI application and which server does it use underneath?",
        "golden_answer": "When you install FastAPI, it comes with a production server, Uvicorn, and you can start it with the fastapi run command.",
        "type": "lookup",
        "source_docs": ["deployment/manually.md"],
        "section": "Use the fastapi run Command",
        "notes": "Verbatim: fastapi run wraps Uvicorn.",
    },
    # ------------------------------------------------------------------
    # multi_hop (12) — the answer combines info from 2 different .md files
    # ------------------------------------------------------------------
    {
        "id": "q021",
        "question": "If an app declares item_id: int as a path parameter and skip=0, limit=10 as query parameters, what does a request to /items/3?skip=20 deliver?",
        "golden_answer": "item_id is converted to the int 3 (path-params: requesting /items/3 returns {\"item_id\":3}); skip is taken from the URL as 20 because you set it in the URL, and limit keeps its default value of 10 (query-params: as they are part of the URL, query values are \"naturally\" strings, then converted to their declared types).",
        "type": "multi_hop",
        "source_docs": ["tutorial/path-params.md", "tutorial/query-params.md"],
        "section": "Path parameters with types + Defaults",
        "notes": "Combine path conversion (path-params) with query defaults/URL override (query-params).",
    },
    {
        "id": "q022",
        "question": "How do you declare a Pydantic request body and then keep a plaintext password out of the response?",
        "golden_answer": "Declare the input model, e.g. UserIn with the password, so FastAPI reads the request body as JSON and validates it (body.md). Then declare an output model such as UserOut without the password and set response_model to it; FastAPI will take care of filtering out all the data that is not declared in the output model (response-model.md).",
        "type": "multi_hop",
        "source_docs": ["tutorial/body.md", "tutorial/response-model.md"],
        "section": "Declare it as a parameter + Add an output model",
        "notes": "Spans Pydantic-body declaration and response model output filtering.",
    },
    {
        "id": "q023",
        "question": "When do background tasks run relative to middleware and to dependency exit code in FastAPI?",
        "golden_answer": "Background tasks run after returning a response (background-tasks.md). Per middleware.md: if you have dependencies with yield, the exit code will run after the middleware; and if there were any background tasks, they will run after all the middleware.",
        "type": "multi_hop",
        "source_docs": ["tutorial/background-tasks.md", "tutorial/middleware.md"],
        "section": "Background Tasks + Technical Details",
        "notes": "Combine the background-tasks definition with the middleware execution-order note.",
    },
    {
        "id": "q024",
        "question": "After creating your first FastAPI app in main.py, how do you write and run a test for it?",
        "golden_answer": "Put the app in main.py (first-steps.md, run with fastapi dev). Then create a TestClient by passing your FastAPI application to it, write functions with names starting with test_ and standard assert statements, install pytest, and run pytest from the terminal — pytest will detect the files and tests automatically (testing.md).",
        "type": "multi_hop",
        "source_docs": ["tutorial/first-steps.md", "tutorial/testing.md"],
        "section": "First Steps + Using TestClient / Run it",
        "notes": "Span the app skeleton and the pytest/TestClient workflow.",
    },
    {
        "id": "q025",
        "question": "How do you declare cookie and header parameters, and what happens if you use a plain parameter instead?",
        "golden_answer": "Declare cookies with Cookie and headers with Header, using the same structure as Path and Query. To declare cookies you need to use Cookie, because otherwise the parameters would be interpreted as query parameters (cookie-params.md); to declare headers you need to use Header for the same reason, and Header converts underscores to hyphens by default (header-params.md).",
        "type": "multi_hop",
        "source_docs": ["tutorial/cookie-params.md", "tutorial/header-params.md"],
        "section": "Declare Cookie parameters + Automatic conversion",
        "notes": "Both files state the plain-parameter -> query-parameter fallback rule.",
    },
    {
        "id": "q026",
        "question": "How do you add string validation to a query parameter and numeric validation to a path parameter?",
        "golden_answer": "Use Query for the query parameter with min_length / max_length / pattern (query-params-str-validations.md), and use Path for the path parameter with gt / ge / lt / le (path-params-numeric-validations.md). Query, Path, and similar classes are subclasses of a common Param class, so they share the same validation arguments.",
        "type": "multi_hop",
        "source_docs": ["tutorial/query-params-str-validations.md", "tutorial/path-params-numeric-validations.md"],
        "section": "Add more validations + Number validations",
        "notes": "Combine string (Query) and numeric (Path) validation declarations.",
    },
    {
        "id": "q027",
        "question": "How does FastAPI combine OAuth2 form login with dependency injection to secure endpoints?",
        "golden_answer": "OAuth2 uses form data for sending the username and password (security/first-steps.md). FastAPI has a powerful Dependency Injection system: path operation functions declare what they require, and FastAPI injects it — this is useful to enforce security and authentication (dependencies/index.md). A dependency such as get_current_user is declared with Depends to protect a path operation.",
        "type": "multi_hop",
        "source_docs": ["tutorial/security/first-steps.md", "tutorial/dependencies/index.md"],
        "section": "The password flow + What is Dependency Injection",
        "notes": "Span OAuth2 first-steps and the dependency injection overview.",
    },
    {
        "id": "q028",
        "question": "Which HTTP status-code ranges represent client errors, and how do you raise your own error status code?",
        "golden_answer": "Status codes from 400 to 499 mean an error from the client (response-status-code.md: an example is 404 for a \"Not Found\" response). To return HTTP responses with errors to the client you use HTTPException — it is a normal Python exception with additional data relevant for APIs, so you raise it, and you can pass any JSON-able value as detail (handling-errors.md).",
        "type": "multi_hop",
        "source_docs": ["tutorial/response-status-code.md", "tutorial/handling-errors.md"],
        "section": "About HTTP status codes + Use HTTPException",
        "notes": "Combine status-code semantics with the HTTPException mechanism.",
    },
    {
        "id": "q029",
        "question": "Can a single FastAPI path operation declare multiple request-body parameters, and how are they encoded?",
        "golden_answer": "Yes: you can add multiple body parameters to your path operation function, e.g. item and user, even though a request can only have a single body (body-multiple-params.md). Request bodies are declared with Pydantic models — FastAPI reads the body of the request as JSON, validates it, and gives you the received data in the parameter (body.md).",
        "type": "multi_hop",
        "source_docs": ["tutorial/body.md", "tutorial/body-multiple-params.md"],
        "section": "Request Body + Multiple body parameters",
        "notes": "Span the Pydantic request-body mechanism and multiple-body declaration.",
    },
    {
        "id": "q030",
        "question": "How would you implement a partial update with PATCH that returns 404 when the item does not exist?",
        "golden_answer": "Use the HTTP PATCH operation to partially update data, and use exclude_unset in item.model_dump(exclude_unset=True) so only the fields actually provided update the stored model (body-updates.md). When the item_id is not found, raise HTTPException with a 404 status code — HTTPException terminates the request right away and sends the HTTP error to the client (handling-errors.md).",
        "type": "multi_hop",
        "source_docs": ["tutorial/body-updates.md", "tutorial/handling-errors.md"],
        "section": "Partial updates with PATCH + Raise an HTTPException in your code",
        "notes": "Combine the PATCH/exclude_unset pattern with HTTPException error handling.",
    },
    {
        "id": "q031",
        "question": "How do you set up a virtual environment for a new FastAPI project and then run the app locally?",
        "golden_answer": "Install uv, create a project with uv init, add fastapi with uv add \"fastapi[standard]\" — uv creates a virtual environment automatically and you run commands with uv run, e.g. uv run fastapi dev (virtual-environments.md). For manual running, use uvicorn main:app, where main is the file main.py and app is the object created with app = FastAPI() (deployment/manually.md).",
        "type": "multi_hop",
        "source_docs": ["virtual-environments.md", "deployment/manually.md"],
        "section": "Create a Project + Run a Server Manually",
        "notes": "Span the uv/venv workflow and the uvicorn run command.",
    },
    {
        "id": "q032",
        "question": "Guiding: how do you run multiple worker processes, and why would a Kubernetes deployment avoid --workers?",
        "golden_answer": "Start multiple workers with the --workers command-line option, e.g. fastapi run --workers 4 main.py, to take advantage of multi-core CPUs (deployment/server-workers.md). But if you have a cluster of machines with Kubernetes, Docker Swarm Mode, or Nomad, you would want to handle replication at the cluster level and run a single Uvicorn process per container instead of multiple workers (deployment/docker.md).",
        "type": "multi_hop",
        "source_docs": ["deployment/server-workers.md", "deployment/docker.md"],
        "section": "Multiple Workers + Build a Docker Image",
        "notes": "Combine the --workers mechanics with the Kubernetes single-process-per-container guidance.",
    },
    # ------------------------------------------------------------------
    # no_answer (12) — plausible for FastAPI, but genuinely absent from corpus
    # ------------------------------------------------------------------
    {
        "id": "q033",
        "question": "How much does it cost to run a FastAPI application on AWS per month?",
        "golden_answer": "Not stated in the corpus. The docs discuss deployment options and FastAPI Cloud, but never publish hosting/pricing costs for third-party clouds like AWS.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "No pricing/hosting cost information exists anywhere in the downloaded docs.",
    },
    {
        "id": "q034",
        "question": "Does FastAPI ship with a built-in rate-limiting middleware?",
        "golden_answer": "Not stated in the corpus. The middleware docs cover custom HTTP middleware and CORS, but no rate limiting is documented.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "Rate limiting is absent from the middleware and advanced-middleware docs.",
    },
    {
        "id": "q035",
        "question": "What are the default settings for FastAPI's built-in user session management?",
        "golden_answer": "Not stated in the corpus. FastAPI has no built-in session management; authentication docs use OAuth2 with username/password but do not define session storage or timeouts.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "Sessions are never mentioned as a framework feature in any doc file.",
    },
    {
        "id": "q036",
        "question": "How do I migrate a FastAPI database schema with Alembic?",
        "golden_answer": "Not stated in the corpus. The SQL databases tutorial uses SQLAlchemy and creates tables directly; Alembic is never mentioned.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "No Alembic / schema-migration content exists in the downloaded docs.",
    },
    {
        "id": "q037",
        "question": "Is a commercial enterprise license available for FastAPI, and how much does it cost?",
        "golden_answer": "Not stated in the corpus. FastAPI is described as open source (MIT), but enterprise licensing options and pricing are never discussed.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "Licensing details beyond open-source/MIT and pricing are not in the docs.",
    },
    {
        "id": "q038",
        "question": "What is the exact maximum size (in MB) before an UploadFile is spilled from memory to disk?",
        "golden_answer": "Not stated in the corpus. The docs say UploadFile uses a spooled file stored in memory up to a maximum size limit, but never give the limit value.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "Only the concept is documented; the specific limit number is absent.",
    },
    {
        "id": "q039",
        "question": "Which database does FastAPI recommend as its production default over all others?",
        "golden_answer": "Not stated in the corpus. The SQL tutorial uses SQLite for the example but the docs explicitly keep the example simple rather than recommending a single production database.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "No single strongest-recommendation database is documented.",
    },
    {
        "id": "q040",
        "question": "How is FastAPI's test suite configured to run on Windows CI runners?",
        "golden_answer": "Not stated in the corpus. The benchmark page and repo links exist, but CI configuration yaml for Windows runners is not part of these docs.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "CI/runner configuration details are not documented in the corpus.",
    },
    {
        "id": "q041",
        "question": "How do I deploy a FastAPI application to Heroku?",
        "golden_answer": "Not stated in the corpus. Deployment docs cover running a server manually, containers/Docker, FastAPI Cloud, and cluster managers, but Heroku is never mentioned.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "Heroku is absent from all deployment documentation.",
    },
    {
        "id": "q042",
        "question": "Which Python version does FastAPI support on Windows ARM64 machines?",
        "golden_answer": "Not stated in the corpus. The docs list Python version requirements generally, but platform-specific (Windows ARM64) support tables are not documented.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "No platform/architecture-specific requirement tables exist.",
    },
    {
        "id": "q043",
        "question": "Can FastAPI be used with Django's ORM as its data layer?",
        "golden_answer": "Not stated in the corpus. The docs cover SQLAlchemy with SQL databases and use Pydantic; Django and its ORM are never mentioned.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "Django is not referenced anywhere in the downloaded docs.",
    },
    {
        "id": "q044",
        "question": "How do I enable mutual TLS (client certificates) for a FastAPI application?",
        "golden_answer": "Not stated in the corpus. TLS/proxy-headers are mentioned in deployment contexts, but mutual TLS with client-certificate verification is not documented.",
        "type": "no_answer",
        "source_docs": [],
        "section": "",
        "notes": "mTLS / client-certificate setup is absent from the corpus.",
    },
    # ------------------------------------------------------------------
    # ambiguous (8) — question is vague, at least two valid interpretations
    # ------------------------------------------------------------------
    {
        "id": "q045",
        "question": "How fast is FastAPI?",
        "golden_answer": "Depends on interpretation: (1) runtime performance — \"Fast: very high performance, on par with NodeJS and Go\"; (2) development speed — \"Fast to code: increase the speed to develop features by about 200% to 300%\". Both readings are valid and retrieve different chunks.",
        "type": "ambiguous",
        "source_docs": ["index.md"],
        "section": "The key features are",
        "notes": "Two readings: benchmark performance vs developer coding speed.",
    },
    {
        "id": "q046",
        "question": "How do I install FastAPI?",
        "golden_answer": "Depends on context: (1) minimal install with uv add fastapi; (2) with the standard extras, uv add \"fastapi[standard]\", which also brings uvicorn/standard and python-multipart for forms/OAuth2. Different docs show each, so multiple answers are valid.",
        "type": "ambiguous",
        "source_docs": ["virtual-environments.md", "tutorial/security/first-steps.md"],
        "section": "Create a Project + Run it",
        "notes": "Minimal vs standard extras install; both are documented.",
    },
    {
        "id": "q047",
        "question": "What do I need python-multipart for?",
        "golden_answer": "Depends on feature: (1) receiving uploaded files, which are sent as form data (request-files.md); (2) OAuth2's password flow, which uses form data to send the username and password (security/first-steps.md). Both are valid and hit different docs.",
        "type": "ambiguous",
        "source_docs": ["tutorial/request-files.md", "tutorial/security/first-steps.md"],
        "section": "Request Files + Run it",
        "notes": "Same package, two different documented reasons.",
    },
    {
        "id": "q048",
        "question": "How do I return an error from my API?",
        "golden_answer": "Depends on the error kind: (1) raise HTTPException with a status code and detail (handling-errors.md); (2) declare a status_code in the path operation decorator so bad results get the right code, using fastapi.status helpers (response-status-code.md). Multiple valid approaches are documented.",
        "type": "ambiguous",
        "source_docs": ["tutorial/handling-errors.md", "tutorial/response-status-code.md"],
        "section": "Use HTTPException + About HTTP status codes",
        "notes": "Exception raising vs declarative status codes.",
    },
    {
        "id": "q049",
        "question": "How do I add parameters to my API?",
        "golden_answer": "Depends on where the data comes from: path parameters (path-params.md), query parameters (query-params.md), headers (header-params.md), cookies (cookie-params.md), or a request body (body.md). Each is declared differently and each answer retrieves different documentation.",
        "type": "ambiguous",
        "source_docs": ["tutorial/query-params.md", "tutorial/header-params.md"],
        "section": "Query Parameters + Header Parameters",
        "notes": "The question does not specify the parameter location.",
    },
    {
        "id": "q050",
        "question": "What happens when an item is not found?",
        "golden_answer": "Depends on the scenario: (1) the handling-errors example raises HTTPException(404) and the client receives {\"detail\": \"Item not found\"}; (2) if the path value cannot be parsed (e.g. /items/foo with item_id typed as int), validation returns an int_parsing error instead. Both error shapes are documented, so the answer is ambiguous.",
        "type": "ambiguous",
        "source_docs": ["tutorial/handling-errors.md", "tutorial/path-params.md"],
        "section": "The resulting response + Data validation",
        "notes": "Application-level 404 vs request-validation error; different chunks.",
    },
    {
        "id": "q051",
        "question": "How do I run my FastAPI application?",
        "golden_answer": "Depends on the stage: (1) development with auto-reload via fastapi dev; (2) production via fastapi run (which wraps Uvicorn) or uvicorn main:app; (3) inside a container with CMD [\"fastapi\", \"run\", \"main.py\", \"--port\", \"80\"]. Multiple valid commands across different docs.",
        "type": "ambiguous",
        "source_docs": ["deployment/manually.md", "tutorial/first-steps.md"],
        "section": "Use the fastapi run Command + First Steps",
        "notes": "Development vs production vs Docker invocation.",
    },
    {
        "id": "q052",
        "question": "What determines the shape of the response my API returns?",
        "golden_answer": "Depends on the mechanism: (1) the path operation function's return type annotation is used for validation, serialization, and filtering (response-model.md); (2) the response_model decorator parameter takes priority and can filter output fields even when the function returns more data; (3) the status_code parameter and response_model_exclude_unset alter the response metadata/content. Several overlapping mechanisms are documented.",
        "type": "ambiguous",
        "source_docs": ["tutorial/response-model.md", "tutorial/response-status-code.md"],
        "section": "Response Model - Return Type + Response Status Code",
        "notes": "Return-type vs response_model vs status_code all shape the response.",
    },
]


def load_golden_dataset() -> List[dict]:
    """Return the 52-question FastAPI golden dataset as dicts (spec schema)."""
    return [dict(q) for q in QUESTIONS]


def _question_from_dict(q: dict) -> GoldenQuestion:
    """Convert a Part-D-schema dict to a :class:`GoldenQuestion`."""
    return GoldenQuestion(
        id=q["id"],
        category=q["type"],
        question=q["question"],
        answer=q["golden_answer"],
        source_documents=list(q.get("source_docs") or []),
        source_sections=[q["section"]] if q.get("section") else [],
        notes=q.get("notes", ""),
    )


class GoldenDataset:
    """Loads and validates the hand-written golden Q&A dataset.

    Provides category filtering and summary statistics so the eval framework
    (Phase 4.2) can run the full suite and per-category breakdowns.
    """

    def __init__(self, questions: List[GoldenQuestion], name: str = "", version: str = ""):
        self.questions = questions
        self.name = name
        self.version = version

    @classmethod
    def from_payload(cls, items, name: str = "", version: str = "") -> "GoldenDataset":
        """Build a dataset from Part-D-schema dicts (e.g. :data:`QUESTIONS`)."""
        return cls(
            questions=[_question_from_dict(q) for q in items],
            name=name,
            version=version,
        )

    @classmethod
    def load(cls, path: str) -> "GoldenDataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("dataset", {})
        questions = [
            GoldenQuestion(**q) for q in data.get("questions", [])
        ]
        return cls(
            questions=questions,
            name=meta.get("name", ""),
            version=meta.get("version", ""),
        )

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def filter(self, category: Optional[str] = None) -> List[GoldenQuestion]:
        if category is None:
            return list(self.questions)
        return [q for q in self.questions if q.category == category]

    def counts(self) -> Dict[str, int]:
        counts = {}
        for cat in CATEGORIES:
            counts[cat] = sum(1 for q in self.questions if q.category == cat)
        counts["total"] = len(self.questions)
        return counts

    def validate(self) -> List[str]:
        """Structural validation. Returns a list of problems (empty if valid)."""
        problems = []
        seen_ids = set()
        for q in self.questions:
            if not q.id:
                problems.append("question with empty id")
            elif q.id in seen_ids:
                problems.append(f"duplicate id: {q.id}")
            seen_ids.add(q.id)

            if q.category not in CATEGORIES:
                problems.append(f"{q.id}: invalid category {q.category!r}")

            if not q.question.strip():
                problems.append(f"{q.id}: empty question")

            if not q.answer.strip():
                problems.append(f"{q.id}: empty answer")

            if q.category in ("lookup", "multi_hop") and not q.source_documents:
                problems.append(f"{q.id}: {q.category} question has no source documents")

            if q.category in ("lookup", "multi_hop") and not q.source_sections:
                problems.append(f"{q.id}: {q.category} question has no source sections")

            if q.category == "no_answer" and q.source_documents:
                problems.append(f"{q.id}: no_answer question lists source documents")

        if len(self.questions) < 50:
            problems.append(f"dataset has {len(self.questions)} questions, expected >= 50")

        for cat in CATEGORIES:
            if cat not in self.counts() or self.counts()[cat] == 0:
                problems.append(f"missing category: {cat}")

        return problems

    def summary(self) -> str:
        counts = self.counts()
        parts = [
            f"Golden Q&A dataset: {self.name} (v{self.version})",
            f"Total questions: {counts['total']}",
        ]
        for cat in CATEGORIES:
            parts.append(f"  - {cat}: {counts[cat]}")
        if self.questions:
            parts.append(f"Covered documents: {sorted({d for q in self.questions for d in q.source_documents})}")
        return "\n".join(parts)