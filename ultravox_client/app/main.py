import os
import json
import time
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# Configuration
GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT", "https://arc.vocallabs.ai/v1/graphql")
GRAPHQL_API_TOKEN = os.getenv("GRAPHQL_API_TOKEN", "legalpwd123")
ULTRAVOX_API_URL = "https://api.ultravox.ai/api/calls"
ULTRAVOX_API_KEY = "b2OAcf3C.4Ej9gFug0688dj5oEA2OWDnX7G9mNPYE" # Needs to be set

# Initialize FastAPI
app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Request/Response ---

class CreateCallRequest(BaseModel):
    agent_id: str
    medium: str = "serverWebSocket" # "serverWebSocket", "sip" or "plivo"
    to_phone: Optional[str] = None
    from_phone: Optional[str] = None # Used for SIP and Plivo outgoing
    sip_username: Optional[str] = None
    sip_password: Optional[str] = None
    plivo_additional_params: Optional[Dict[str, Any]] = None

# New models for Scheduled Call Batches
class PlivoOutgoingMedium(BaseModel):
    to: str
    from_: str = Field(..., alias="from") # 'from' is a Python keyword, so use an alias
    additionalParams: Optional[Dict[str, Any]] = None

class SipOutgoingMedium(BaseModel):
    to: str
    from_: str = Field(..., alias="from")
    username: Optional[str] = None
    password: Optional[str] = None

class ScheduledCallMediumConfig(BaseModel):
    plivo: Optional[PlivoOutgoingMedium] = None
    sip: Optional[SipOutgoingMedium] = None
    # Add other mediums as needed, e.g., twilio, serverWebSocket

class ScheduledCall(BaseModel):
    medium: ScheduledCallMediumConfig
    metadata: Optional[Dict[str, Any]] = None
    templateContext: Optional[Dict[str, Any]] = None
    experimentalSettings: Optional[Dict[str, Any]] = None

class CreateScheduledBatchRequest(BaseModel):
    windowStart: Optional[datetime] = None
    windowEnd: Optional[datetime] = None
    webhookUrl: Optional[str] = None
    webhookSecret: Optional[str] = None
    paused: bool = False
    calls: List[ScheduledCall]

# --- GraphQL ---

def execute_graphql(query: str, variables: Dict[str, Any] = None):
    headers = {
        "Content-Type": "application/json",
        "x-hasura-admin-secret": "legalpwd123",
    }
    response = requests.post(
        GRAPHQL_ENDPOINT,
        json={"query": query, "variables": variables},
        headers=headers,
    )
    response.raise_for_status()
    return response.json()

GET_AGENT_QUERY = """
query GetAgent($id: uuid!) {
  vocallabs_agent(where: {id: {_eq: $id}}) {
    agent_prompt
    name
    welcome_message
    temperature
    ai_model {
      provider
      model
    }
    actions {
      base_action {
        action_name
        description
        external_curl
        action_request_method
        parameters {
          description
          name
          type
          is_required
        }
      }
    }
    data_collection
    knowledge_base_vector_count
  }
}
"""

# --- Ultravox Client ---

def create_call_with_retry(call_config: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": ULTRAVOX_API_KEY
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(ULTRAVOX_API_URL, json=call_config, headers=headers)
            
            if response.status_code == 429: # Too Many Requests
                retry_after = response.headers.get('Retry-After')
                delay = 1
                if retry_after:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        # Parse HTTP Date format if not integer
                        try:
                            delay = (parsedate_to_datetime(retry_after) - datetime.utcnow()).total_seconds()
                        except Exception:
                            pass # Default delay
                
                logger.warning(f"Rate limited. Retrying in {delay:.1f} seconds...")
                time.sleep(max(1, delay)) # Ensure at least 1 sec
                continue
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt) # Exponential backoff for other errors

    raise Exception("Max retries reached")

# --- Logic ---

def map_agent_to_ultravox_config(agent: Dict[str, Any], request: CreateCallRequest) -> Dict[str, Any]:
    # 1. Basic Config
    config = {
        "systemPrompt": agent.get("agent_prompt", "You are a helpful assistant."),
        "model": "fixie-ai/ultravox", # Or map from agent['ai_model'] if compatible
        "temperature": agent.get("temperature", 0.7),
        "metadata": {} # Initialize metadata
    }

    if agent.get("welcome_message"):
        config["firstSpeakerMessage"] = agent["welcome_message"]

    # Add RequiredValues to metadata
    data_collection = agent.get("data_collection")
    if data_collection and isinstance(data_collection, dict):
        required_values = data_collection.get("required_values", {})
        if required_values:
            metadata_from_required = {}
            for key, val in required_values.items():
                if isinstance(val, dict) and val.get("description"):
                    metadata_from_required[key] = val["description"]
                else:
                    metadata_from_required[key] = str(val) # Fallback if no description or not dict
            config["metadata"].update(metadata_from_required)

    # 2. Medium (SIP or WebSocket)
    if request.medium == "sip":
        if not request.to_phone:
             raise HTTPException(status_code=400, detail="to_phone is required for SIP")
        
        config["medium"] = {
            "sip": {
                "outgoing": {
                    "to": request.to_phone,
                    "from": request.from_phone or "unknown",
                    "username": request.sip_username,
                    "password": request.sip_password
                }
            }
        }
    elif request.medium == "plivo":
        if not request.to_phone or not request.from_phone:
             raise HTTPException(status_code=400, detail="to_phone and from_phone are required for Plivo")
        
        config["medium"] = {
            "plivo": {
                "outgoing": {
                    "to": request.to_phone,
                    "from": request.from_phone,
                    "additionalParams": request.plivo_additional_params or {}
                }
            }
        }
    else:
        # Default to WebSocket
        config["medium"] = {
            "serverWebSocket": {
                "inputSampleRate": 48000,
                "outputSampleRate": 48000,
                "clientBufferSizeMs": 60
            }
        }

    # 3. Tools
    tools = []
    
    # Custom Tools (Actions)
    actions = agent.get("actions", [])
    for action_wrapper in actions:
        base_action = action_wrapper.get("base_action", {})
        if not base_action:
            continue
            
        # Map parameters to Dynamic Parameters Schema
        dynamic_params = []
        raw_params = base_action.get("parameters", [])
        # Handle if parameters is stored as string (JSON) or list/dict
        if isinstance(raw_params, str):
            try:
                raw_params = json.loads(raw_params)
            except:
                raw_params = []
        
        for p in raw_params:
             dynamic_params.append({
                 "name": p.get("name"),
                 "location": "PARAMETER_LOCATION_BODY",
                 "required": p.get("is_required", False),
                 "schema": {
                     "type": p.get("type", "string"),
                     "description": p.get("description", "")
                 }
             })

        tool_def = {
            "modelToolName": base_action.get("action_name"),
            "description": base_action.get("description"),
            "dynamicParameters": dynamic_params,
            "http": {
                "baseUrlPattern": base_action.get("external_curl"),
                "httpMethod": base_action.get("action_request_method", "POST")
            }
        }
        tools.append(tool_def)

    # Call Cut (HangUp)
    data_collection = agent.get("data_collection")
    if data_collection and isinstance(data_collection, dict):
        call_cuts = data_collection.get("call_cut", [])
        if call_cuts:
             # Enable built-in hangUp tool
             tools.append({"toolName": "hangUp"})

    # RAG (queryCorpus)
    if agent.get("knowledge_base_vector_count", 0) > 0:
        # You need to map Agent ID to Corpus ID. This is a placeholder.
        # In a real scenario, you might have a direct field in agent data for corpus_id
        # or a lookup mechanism.
        corpus_id = os.getenv("ULTRAVOX_DEFAULT_CORPUS_ID", f"corpus_{agent.get('name', 'default')}")
        logger.info(f"Adding queryCorpus tool with corpus_id: {corpus_id}")
        tools.append({
            "toolName": "queryCorpus",
            "parameterOverrides": {
                "corpus_id": corpus_id,
                "max_results": 5 # Example default, can be dynamic
            }
        })

    if tools:
        config["selectedTools"] = tools

    return config

# --- Post-Call Data Collection ---

def get_call_details(call_id: str) -> Dict[str, Any]:
    base_url = "https://api.ultravox.ai/api/calls"
    headers = {
        "X-API-Key": ULTRAVOX_API_KEY
    }

    call_detail_url = f"{base_url}/{call_id}"
    messages_url = f"{base_url}/{call_id}/messages"
    recording_url = f"{base_url}/{call_id}/recording" # This returns raw audio, not URL directly

    details = {}
    try:
        call_resp = requests.get(call_detail_url, headers=headers)
        call_resp.raise_for_status()
        details["call_info"] = call_resp.json()
        details["end_reason"] = details["call_info"].get("endReason") # Extract endReason
        details["sip_termination_reason"] = details["call_info"].get("sipDetails", {}).get("terminationReason") # Extract SIP termination reason
        details["call_summary"] = details["call_info"].get("summary")
        details["call_metadata"] = details["call_info"].get("metadata")

    except requests.RequestException as e:
        logger.error(f"Failed to get call info for {call_id}: {e}")
        details["call_info_error"] = str(e)

    try:
        messages_resp = requests.get(messages_url, headers=headers)
        messages_resp.raise_for_status()
        details["messages"] = messages_resp.json()
    except requests.RequestException as e:
        logger.error(f"Failed to get messages for {call_id}: {e}")
        details["messages_error"] = str(e)
    
    # Note: Recording endpoint returns audio data directly, not a URL.
    # To get a persistent URL, you'd typically save this data to a storage service
    # and return its URL. For now, we'll just acknowledge its existence.
    details["recording_available_info"] = f"Access recording at {recording_url} (requires direct download)"

    return details

@app.post("/call-details")
async def get_call_details_endpoint(call_id: str = Body(..., embed=True)):
    if not ULTRAVOX_API_KEY:
        raise HTTPException(status_code=500, detail="ULTRAVOX_API_KEY not configured")
    
    try:
        details = get_call_details(call_id)
        return {"status": "success", "data": details}
    except Exception as e:
        logger.error(f"Error fetching call details for {call_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch call details: {str(e)}")

@app.post("/create-call")
async def create_call_endpoint(request: CreateCallRequest):
    if not ULTRAVOX_API_KEY:
        raise HTTPException(status_code=500, detail="ULTRAVOX_API_KEY not configured")

    # 1. Fetch Agent
    try:
        logger.info(f"Attempting to fetch agent with ID: {request.agent_id}")
        data = execute_graphql(GET_AGENT_QUERY, {"id": request.agent_id})
        
        agents = data.get("data", {}).get("vocallabs_agent")
        if not agents:
            logger.warning(f"Agent with ID {request.agent_id} not found. GraphQL response: {data}")
            raise HTTPException(status_code=404, detail=f"Agent with ID '{request.agent_id}' not found")
        agent = agents[0]
    except Exception as e:
        logger.error(f"GraphQL Error fetching agent {request.agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch agent '{request.agent_id}': {str(e)}")

    # 2. Map Config
    try:
        ultravox_config = map_agent_to_ultravox_config(agent, request)
    except Exception as e:
         logger.error(f"Mapping Error: {e}")
         raise HTTPException(status_code=500, detail=f"Failed to map config: {str(e)}")

    # 3. Create Call
    try:
        call_response = create_call_with_retry(ultravox_config)
        return call_response
    except Exception as e:
        logger.error(f"Ultravox API Error: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to create Ultravox call: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/agents/{agent_id}/scheduled-batches")
async def create_scheduled_batch_endpoint(
    agent_id: str,
    request: CreateScheduledBatchRequest
):
    if not ULTRAVOX_API_KEY:
        raise HTTPException(status_code=500, detail="ULTRAVOX_API_KEY not configured")

    ultravox_scheduled_batches_url = f"https://api.ultravox.ai/api/agents/{agent_id}/scheduled_batches"
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": ULTRAVOX_API_KEY
    }

    try:
        # Convert Pydantic model to dictionary, handling aliases and Optional fields
        payload = request.dict(by_alias=True, exclude_none=True)
        
        # Manually handle from_ alias in nested medium config if present
        for call in payload.get("calls", []):
            medium_config = call.get("medium")
            if medium_config:
                if "plivo" in medium_config and medium_config["plivo"] and "from_" in medium_config["plivo"]:
                    medium_config["plivo"]["from"] = medium_config["plivo"].pop("from_")
                if "sip" in medium_config and medium_config["sip"] and "from_" in medium_config["sip"]:
                    medium_config["sip"]["from"] = medium_config["sip"].pop("from_")

        logger.info(f"Sending scheduled batch request to Ultravox for agent {agent_id}. Payload: {json.dumps(payload)}")
        
        response = requests.post(
            ultravox_scheduled_batches_url,
            json=payload,
            headers=headers
        )
        response.raise_for_status() # Raise an exception for HTTP errors
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating scheduled batch for agent {agent_id}: {e}. Response: {e.response.text if e.response else 'No response'}")
        raise HTTPException(status_code=e.response.status_code if e.response else 500, detail=f"Failed to create scheduled batch: {e.response.text if e.response else str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating scheduled batch for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
