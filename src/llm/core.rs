pub trait LLMClient {
    fn set_api_key(&mut self, api_key: &str);
    fn get_api_key(&self) -> &str;
    async fn create_completion(&self, messages: &dyn Messages) -> String;
}

pub trait Message {
    fn role(&self) -> &str;
    fn content(&self) -> &str;
}

pub trait Messages {
    fn add_message(&mut self, role: &str, content: &str);
    fn list_messages(&self) -> Vec<Box<dyn Message>>;
}

pub trait Model {
    fn to_string(&self) -> String;
}