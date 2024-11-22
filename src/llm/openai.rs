use reqwest;
use serde::Serialize;
use crate::llm::core::*;

#[derive(Default, Serialize)]
pub enum OpenAIModel {
    #[default]
    GPT4oMini,
    GPT4o,
}

impl Model for OpenAIModel {
    fn to_string(&self) -> String {
        match self {
            OpenAIModel::GPT4oMini => "gpt-4o-mini".to_string(),
            OpenAIModel::GPT4o => "gpt-4o".to_string(),
        }
    }
}

#[derive(Serialize, Clone)]
struct OpenAIMessage {
    role: String,
    content: String,
}

impl OpenAIMessage {
    pub fn from(message: (&str, &str)) -> Self {
        Self {
            role: message.0.to_string(),
            content: message.1.to_string(),
        }
    }
}

impl Message for OpenAIMessage {
    fn role(&self) -> &str {
        &self.role
    }
    fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Serialize, Clone)]
pub struct OpenAIMessages {
    messages: Vec<OpenAIMessage>,
}

impl OpenAIMessages {
    pub fn from(messages: Vec<(&str, &str)>) -> Self {
        let formatted_messages = messages.into_iter().map(OpenAIMessage::from).collect();
        Self {
            messages: formatted_messages,
        }
    }
}

impl Messages for OpenAIMessages {
    fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(OpenAIMessage::from((role, content)));
    }

    fn list_messages(&self) -> Vec<Box<dyn Message>> {
        self.messages
            .iter()
            .map(|msg| Box::new(msg.clone()) as Box<dyn Message>)
            .collect()
    }
}

#[derive(Serialize)]
pub struct OpenAICompletionRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
}

impl OpenAICompletionRequest {
    fn new(model: String, messages: OpenAIMessages) -> Self {
        Self {
            model,
            messages: messages.messages,
        }
    }
}

pub struct OpenAIClient {
    model: OpenAIModel,
    completion_endpoint: String,
    api_key: String,
}

impl OpenAIClient {
    pub fn new(model: OpenAIModel, api_key: String) -> Self {
        Self {
            model,
            completion_endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key,
        }
    }
}

impl LLMClient for OpenAIClient {
    fn set_api_key(&mut self, api_key: &str) {
        self.api_key = api_key.to_string();
    }

    fn get_api_key(&self) -> &str {
        &self.api_key
    }

    async fn create_completion(&self, messages: &dyn Messages) -> String {
        let openai_messages = OpenAIMessages {
            messages: messages
                .list_messages()
                .iter()
                .map(|msg| OpenAIMessage {
                    role: msg.role().to_string(),
                    content: msg.content().to_string(),
                })
                .collect(),
        };
        let request = OpenAICompletionRequest::new(self.model.to_string(), openai_messages);
        let payload = match serde_json::to_string(&request) {
            Ok(p) => p,
            Err(e) => return format!("Error serializing request: {}", e),
        };
        let web_client = reqwest::Client::new();
        let response = web_client
            .post(&self.completion_endpoint)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .body(payload)
            .send()
            .await;

        match response {
            Ok(res) => {
                if res.status().is_success() {
                    res.text().await.unwrap_or_else(|_| "Failed to read response".to_string())
                } else {
                    format!(
                        "Error: {} {}",
                        res.status(),
                        res.text().await.unwrap_or_else(|_| "Failed to read response".to_string())
                    )
                }
            }
            Err(e) => format!("Request failed: {}", e),
        }
    }
}
