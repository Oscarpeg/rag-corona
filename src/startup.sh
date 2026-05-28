#!/bin/bash
echo "Verificando Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Iniciando Ollama..."
    ollama serve &
    sleep 6
fi
echo "Ollama listo. Modelos:" && ollama list
