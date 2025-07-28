from setuptools import setup, find_packages

setup(
    name="coach-critique",
    version="0.1.0",
    description="Safety-First Coaching with Bounded LLM & Evidence Gate",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "rank-bm25>=0.2.2",
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "aiohttp>=3.8.5",
        "httpx>=0.24.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=13.0.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "coach-critique=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 