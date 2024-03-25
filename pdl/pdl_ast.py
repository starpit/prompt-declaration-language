from enum import StrEnum
from typing import Any, Literal, Optional, TypeAlias

from genai.schema import (
    ModerationParameters,
    PromptTemplateData,
    TextGenerationParameters,
)
from pydantic import BaseModel, ConfigDict, Field, RootModel

ScopeType: TypeAlias = dict[str, Any]

ExpressionType: TypeAlias = Any
# (
#     str
#     | int
#     | float
#     | bool
#     | None
#     | list["ExpressionType"]
#     | dict[str, "ExpressionType"]
# )


class BlockKind(StrEnum):
    FUNCTION = "function"
    CALL = "call"
    MODEL = "model"
    CODE = "code"
    API = "api"
    GET = "get"
    DATA = "data"
    DOCUMENT = "document"
    IF = "if"
    REPEAT = "repeat"
    REPEAT_UNTIL = "repeat_until"
    READ = "read"
    INCLUDE = "include"
    ERROR = "error"
    EMPTY = "empty"
    FOR = "for"


class Block(BaseModel):
    """PDL program block"""

    model_config = ConfigDict(extra="forbid")
    description: Optional[str] = None
    spec: Any = None  # TODO
    defs: dict[str, "BlocksType"] = {}
    assign: Optional[str] = Field(default=None, alias="def")
    show_result: bool = True
    result: Optional[Any] = None


class FunctionBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.FUNCTION] = BlockKind.FUNCTION
    function: Optional[dict[str, Any]]
    returns: "BlocksType" = Field(..., alias="return")
    scope: Optional[ScopeType] = None


class CallBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.CALL] = BlockKind.CALL
    call: str
    args: dict[str, Any] = {}


class PDLTextGenerationParameters(TextGenerationParameters):
    model_config = ConfigDict(extra="forbid")


class ModelBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.MODEL] = BlockKind.MODEL
    model: str
    input: Optional["BlocksType"] = None
    prompt_id: Optional[str] = None
    parameters: Optional[PDLTextGenerationParameters] = None
    moderations: Optional[ModerationParameters] = None
    data: Optional[PromptTemplateData] = None
    constraints: Any = None  # TODO


class CodeBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.CODE] = BlockKind.CODE
    lan: str
    code: "BlocksType"


class ApiBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.API] = BlockKind.API
    api: str
    url: str
    input: "BlocksType"


class GetBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.GET] = BlockKind.GET
    get: str


class DataBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.DATA] = BlockKind.DATA
    data: ExpressionType


class DocumentBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.DOCUMENT] = BlockKind.DOCUMENT
    document: "BlocksType"


class IfBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.IF] = BlockKind.IF
    condition: ExpressionType = Field(alias="if")
    then: "BlocksType"
    elses: Optional["BlocksType"] = Field(default=None, alias="else")
    if_result: Optional[bool] = None


class ForBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.FOR] = BlockKind.FOR
    fors: dict[str, Any] = Field(alias="for")
    repeat: "BlocksType"
    trace: list["BlocksType"] = []


class RepeatBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.REPEAT] = BlockKind.REPEAT
    repeat: "BlocksType"
    num_iterations: int
    trace: list["BlocksType"] = []


class RepeatUntilBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.REPEAT_UNTIL] = BlockKind.REPEAT_UNTIL
    repeat: "BlocksType"
    until: ExpressionType
    trace: list["BlocksType"] = []


class ReadBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.READ] = BlockKind.READ
    read: str | None
    message: Optional[str] = None
    multiline: bool = False
    parser: Optional[str] = None  # json


class IncludeBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.INCLUDE] = BlockKind.INCLUDE
    include: str
    trace: Optional["BlockType"] = None


class ErrorBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.ERROR] = BlockKind.ERROR
    msg: str
    program: "BlocksType"


class EmptyBlock(Block):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[BlockKind.EMPTY] = BlockKind.EMPTY


AdvancedBlockType: TypeAlias = (
    FunctionBlock
    | CallBlock
    | ModelBlock
    | CodeBlock
    | ApiBlock
    | GetBlock
    | DataBlock
    | IfBlock
    | RepeatBlock
    | RepeatUntilBlock
    | ForBlock
    | DocumentBlock
    | ReadBlock
    | IncludeBlock
    | ErrorBlock
    | EmptyBlock
)

BlockType: TypeAlias = str | AdvancedBlockType
BlocksType: TypeAlias = BlockType | list[BlockType]  # pyright: ignore


class Program(RootModel):
    """
    Prompt Description Program (PDL)
    """

    # root: dict[str, BlockType]
    root: BlockType


class PdlBlock(RootModel):
    # This class is used to introduce that a type in the generate JsonSchema
    root: BlockType


class PdlBlocks(RootModel):
    # This class is used to introduce that a type in the generate JsonSchema
    root: BlocksType
