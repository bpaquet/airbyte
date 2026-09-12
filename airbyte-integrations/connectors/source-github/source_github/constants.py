#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

TOKEN_SEPARATOR = ","
# GitHub's REST API max. Was 10: at that size, a repository with enough history in a
# "large stream" (issues, comments, ...) reaches enough pages to hit GitHub's page-depth cutoff
# ("Pagination with the page parameter is not supported for large datasets"), failing the sync
# outright. 100 cuts the page count (and the odds of hitting that cutoff) by 10x.
DEFAULT_PAGE_SIZE_FOR_LARGE_STREAM = 100
DEFAULT_PAGE_SIZE = 100
PERSONAL_ACCESS_TOKEN_TITLE = "Personal Access Token"
ACCESS_TOKEN_TITLE = "Access Token"
GITHUB_APP_TITLE = "GitHub App"
