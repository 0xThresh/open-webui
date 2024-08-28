"""
title: Confluence Pipe
author: 0xThresh
author_url: https://github.com/0xthresh
funding_url: https://github.com/open-webui
version: 0.1
license: MIT
requirements: atlassian-python-api, pytesseract, Pillow
"""

import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from utils.misc import get_last_user_message


class Pipe:
    class Valves(BaseModel):
        CONFLUENCE_SITE: str = Field(
            default="",
            description="The URL of your Confluence site, such as https://open-webui.atlassian.net/",
        )
        CONFLUENCE_SPACE: str = Field(
            default="",
            description="The Confluence space to pull pages from",
        )
        CONFLUENCE_USERNAME: str = Field(
            default="",
            description="Your Confluence username for the site you pull documents from",
        )
        CONFLUENCE_API_KEY: str = Field(
            default="",
            description="Your Confluence API key, generated here for Confluence Cloud: https://id.atlassian.com/manage-profile/security/api-tokens",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "Confluence RAG"
        self.name = "confluence_rag"
        self.valves = self.Valves(
            **{
                "CONFLUENCE_SITE": os.getenv("CONFLUENCE_SITE", ""),
                "CONFLUENCE_SPACE": os.getenv("CONFLUENCE_SPACE", ""),
                "CONFLUENCE_USERNAME": os.getenv("CONFLUENCE_USERNAME", ""),
                "CONFLUENCE_API_KEY": os.getenv("CONFLUENCE_API_KEY", ""),
            }
        )
        pass

    def get_confluence_docs(self):
        loader = ConfluenceLoader(
            url=self.valves.CONFLUENCE_SITE,
            username=self.valves.CONFLUENCE_USERNAME,
            api_key=self.valves.CONFLUENCE_API_KEY,
            space_key=self.valves.CONFLUENCE_SPACE,
            include_attachments=True,
            # Start by limiting to 5 to avoid overwhelming the context window
            limit=5,
        )
        documents = loader.load()
        return documents

    def pipes(self):
        return [
            {
                "id": self.id,
                "name": self.name,
            }
        ]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        # Get the documents from Confluence
        documents = self.get_confluence_docs()

        # split docs into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Load documents into Chroma
        db = Chroma.from_documents(docs, embedding_function)
        print("---------DB----------")
        print(db)
        print(type(db))

        # Get the query from the user and query Chroma with it
        query = get_last_user_message(body["messages"])
        print("------QUERY------")
        print(query)
        docs = db.similarity_search(query)

        # Return response
        # print(docs[0].page_content)
        # response = query_engine.query(user_message)
        # print(docs)

        return docs[0].page_content
