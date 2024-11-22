#![allow(dead_code)]
mod llm;
use llm::*;
use std::io::{self, Write};

#[tokio::main]
async fn main() {
    let mut messages: OpenAIMessages = OpenAIMessages::from(vec![
        ("system", "Eres un adivino mistico, con respuestas cripticas!"),
        ("assistant", "Si amo ese sere yo!"),
    ]);

    let client = OpenAIClient::new(
        OpenAIModel::GPT4o,
        "".to_string()
    );

    conversation_loop(client, &mut messages).await;

    // let response = client.create_completion(messages).await;
    // println!("Response: {}", response);
}

async fn conversation_loop(client: OpenAIClient, messages_buffer: &mut OpenAIMessages) {
    loop {
        print!("user@conversation $ ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();

        match io::stdin().read_line(&mut user_input) {
            Ok(_) => {
                let user_input = user_input.trim().to_string();

                if user_input.eq_ignore_ascii_case("!exit") {
                    println!("Adiós!");
                    break;
                }

                messages_buffer.add_message("user", &user_input);
                
                let response = client.create_completion(messages_buffer).await;
                println!("Response: {}", response);

                
            }
            Err(error) => {
                println!("Error al leer la entrada: {}", error);
            }
        }
    }
}
