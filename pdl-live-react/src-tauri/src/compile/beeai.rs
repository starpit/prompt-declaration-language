use ::std::collections::HashMap;
use ::std::error::Error;
use ::std::fs::File;
use ::std::io::BufReader;

use serde::{Deserialize, Serialize};
use serde_json::{from_reader, json, to_string, Value};

macro_rules! zip {
    ($x: expr) => ($x);
    ($x: expr, $($y: expr), +) => (
        $x.into_iter().zip(
            zip!($($y), +))
    )
}

#[derive(Deserialize, Debug)]
struct BeeAiInputStateDict {
    prompt: Option<String>,
    // expected_output: Option<String>,
}
#[derive(Deserialize, Debug)]
struct BeeAiInputState {
    #[serde(rename = "__dict__")]
    dict: BeeAiInputStateDict,
}
#[derive(Deserialize, Debug)]
struct BeeAiInput {
    #[serde(rename = "py/state")]
    state: BeeAiInputState,
}
/*#[derive(Deserialize, Debug)]
struct JsonSchemaParameter {
    #[serde(rename = "type")]
    parameter_type: String,
    description: String,
    title: String,
}*/
#[derive(Deserialize, Debug)]
struct BeeAiToolSchema {
    properties: HashMap<String, Value>,
}
#[derive(Deserialize, Debug)]
struct BeeAiToolState {
    name: String,
    description: Option<String>,
    input_schema: BeeAiToolSchema,
    options: Option<HashMap<String, Value>>,
}
#[derive(Deserialize, Debug)]
struct BeeAiTool {
    #[serde(rename = "py/state")]
    state: BeeAiToolState,
}
#[derive(Deserialize, Debug)]
struct BeeAiLlmParametersState {
    #[serde(rename = "__dict__")]
    dict: HashMap<String, Value>,
}
#[derive(Deserialize, Debug)]
struct BeeAiLlmParameters {
    #[serde(rename = "py/state")]
    state: BeeAiLlmParametersState,
}
#[derive(Deserialize, Debug)]
struct BeeAiLlmSettings {
    api_key: String,
    // base_url: String,
}
#[derive(Deserialize, Debug)]
struct BeeAiLlm {
    // might be helpful to know it's Ollama?
    //#[serde(rename = "py/object")]
    //object: String,
    parameters: BeeAiLlmParameters,

    #[serde(rename = "_model_id")]
    model_id: String,
    //#[serde(rename = "_litellm_provider_id")]
    //provider_id: String,
    #[serde(rename = "_settings")]
    settings: BeeAiLlmSettings,
}
#[derive(Deserialize, Debug)]
struct BeeAiWorkflowStepStateMeta {
    //name: String,
    role: String,
    llm: BeeAiLlm,
    instructions: Option<String>,
    tools: Option<Vec<BeeAiTool>>,
}
#[derive(Deserialize, Debug)]
struct BeeAiWorkflowStepStateDict {
    meta: BeeAiWorkflowStepStateMeta,
}
#[derive(Deserialize, Debug)]
struct BeeAiWorkflowStepState {
    #[serde(rename = "__dict__")]
    dict: BeeAiWorkflowStepStateDict,
}
#[derive(Deserialize, Debug)]
struct BeeAiWorkflowStep {
    #[serde(rename = "py/state")]
    state: BeeAiWorkflowStepState,
}
#[derive(Deserialize, Debug)]
struct BeeAiWorkflowInner {
    #[serde(rename = "_name")]
    name: String,
    #[serde(rename = "_steps")]
    steps: HashMap<String, BeeAiWorkflowStep>,
}
#[derive(Deserialize, Debug)]
struct BeeAiWorkflow {
    workflow: BeeAiWorkflowInner,
}
#[derive(Deserialize, Debug)]
struct BeeAiProgram {
    inputs: Vec<BeeAiInput>,
    workflow: BeeAiWorkflow,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
enum PdlBlock {
    String(String),
    Text {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        text: Vec<PdlBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        defs: Option<HashMap<String, PdlBlock>>,
    },
    Model {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        model: String,
        parameters: HashMap<String, Value>,
    },
    PythonFunction {
        lang: String,
        code: String,
    },
}
/*#[derive(Serialize, Debug)]
struct PdlFunctionJsonSchemaParameters {
    #[serde(rename = "type")]
    schema_type: String, // TODO constant "object"
    properties: Option<HashMap<String, Value>>,
}
#[derive(Serialize, Debug)]
struct PdlFunctionJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: PdlFunctionJsonSchemaParameters,
}
#[derive(Serialize, Debug)]
struct PdlFunctionDeclaration {
    #[serde(rename = "type")]
    declaration_type: String, // TODO constant "function"
    function: PdlFunctionJsonSchema,
}*/

fn a_tool(tool: &BeeAiToolState) -> Value {
    json!({
        "type": "function",
        "function": json!({
            "name": tool.name,
            "description": tool.description,
            "parameters": json!({
                "type": "object",
                "properties": tool.input_schema.properties,
            }),
            "options": tool.options
        })
    })
}

fn with_tools(
    tools: &Option<Vec<BeeAiTool>>,
    parameters: HashMap<String, Value>,
) -> HashMap<String, Value> {
    match tools {
        Some(tools) => {
            let mut copy = parameters.clone();
            copy.insert(
                "tools".to_string(),
                tools.into_iter().map(|tool| a_tool(&tool.state)).collect(),
            );
            copy
        }
        _ => parameters,
    }
}

fn call_tools(tools: &Vec<BeeAiTool>) -> PdlBlock {
    PdlBlock::Text {
        defs: None,
        role: None,
        description: Some(format!(
            "Calling tools {:?}",
            tools
                .into_iter()
                .map(|tool| &tool.state.name)
                .collect::<Vec<_>>()
        )),
        text: tools
            .into_iter()
            .map(|tool| PdlBlock::String(format!("${{ pdl__tool_{} }}", tool.state.name)))
            .collect(),
    }
}

pub fn compile(source_file_path: &String, output_path: &String) -> Result<(), Box<dyn Error>> {
    println!("Compiling beeai {} to {}", source_file_path, output_path);

    // Open the file in read-only mode with buffer.
    let file = File::open(source_file_path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let bee: BeeAiProgram = from_reader(reader)?;

    let inputs: Vec<PdlBlock> = bee
        .inputs
        .into_iter()
        .map(|input| input.state.dict.prompt)
        .flatten()
        .map(|prompt| PdlBlock::String(format!("{}\n", prompt)))
        .collect::<Vec<_>>();

    let system_prompts = bee
        .workflow
        .workflow
        .steps
        .values()
        .filter_map(|step| step.state.dict.meta.instructions.clone())
        .map(|instructions| PdlBlock::Text {
            role: Some(String::from("system")),
            text: vec![PdlBlock::String(instructions)],
            defs: None,
            description: None,
        })
        .collect::<Vec<_>>();

    let tool_declarations = bee
        .workflow
        .workflow
        .steps
        .values()
        .filter_map(|step| step.state.dict.meta.tools.as_ref())
        .flat_map(|tools| {
            tools
                .into_iter()
                .map(|BeeAiTool { state }| {
                    (
                        format!("pdl__tool_{}", state.name),
                        PdlBlock::PythonFunction {
                            lang: "python".to_string(),
                            code: "result = 'hello'".to_string(),
                        },
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<HashMap<_, _>>();

    let model_calls = bee
        .workflow
        .workflow
        .steps
        .into_values()
        .map(|step| {
            (
                step.state.dict.meta.role,
                step.state.dict.meta.tools,
                step.state.dict.meta.llm,
            )
        })
        .map(|(role, tools, llm)| {
            let model_block = PdlBlock::Model {
                description: Some(role),
                model: format!("{}/{}", llm.settings.api_key, llm.model_id),
                parameters: with_tools(&tools, llm.parameters.state.dict),
            };
            match tools {
                Some(tools) => vec![model_block, call_tools(&tools)],
                None => vec![model_block],
            }
        })
        .collect::<Vec<_>>();
    println!("!!!!!!!!!!!! {:?}", model_calls);

    let body = zip!(inputs, system_prompts, model_calls)
        .map(|(a, (b, c))| {
            let mut v = vec![a, b];
            v.extend(c);
            v
        })
        .flatten()
        .collect::<Vec<_>>();

    let pdl: PdlBlock = PdlBlock::Text {
        defs: if tool_declarations.len() == 0 {
            None
        } else {
            Some(tool_declarations)
        },
        description: Some(bee.workflow.workflow.name),
        role: None,
        text: body,
    };

    match output_path.as_str() {
        "-" => println!("{}", to_string(&pdl)?),
        _ => {
            ::std::fs::write(output_path, to_string(&pdl)?)?;
        }
    }

    Ok(())
}
