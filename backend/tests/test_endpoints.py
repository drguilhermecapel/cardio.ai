#!/usr/bin/env python3
"""
Script para testar todos os endpoints do CardioAI Pro
Execute este script para verificar quais endpoints estão disponíveis
"""

import requests
import json
from typing import Dict, List, Tuple

def test_endpoint(url: str, method: str = "GET") -> Tuple[int, Dict]:
    """Testa um endpoint e retorna status code e resposta."""
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.request(method, url, timeout=5)
        
        try:
            data = response.json()
        except:
            data = {"text": response.text[:200]}
        
        return response.status_code, data
    except requests.exceptions.ConnectionError:
        return 0, {"error": "Connection refused - servidor não está rodando"}
    except requests.exceptions.Timeout:
        return 0, {"error": "Timeout - servidor não respondeu"}
    except Exception as e:
        return 0, {"error": str(e)}

def main():
    """Testa todos os endpoints conhecidos do CardioAI Pro."""
    
    print("🔍 CardioAI Pro - Teste de Endpoints")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Lista de endpoints para testar
    endpoints = [
        # Endpoints básicos
        ("/", "Root"),
        ("/health", "Health Check"),
        
        # Documentação sem prefixo
        ("/docs", "Swagger UI (sem prefixo)"),
        ("/redoc", "ReDoc (sem prefixo)"),
        ("/openapi.json", "OpenAPI Schema (sem prefixo)"),
        
        # Documentação com prefixo api/v1
        ("/api/v1/docs", "Swagger UI (com prefixo)"),
        ("/api/v1/redoc", "ReDoc (com prefixo)"),
        ("/api/v1/openapi.json", "OpenAPI Schema (com prefixo)"),
        
        # Endpoints da API
        ("/api/v1/", "API Root"),
        ("/api/v1/health", "API Health"),
        ("/api/v1/auth/login", "Auth Login"),
        ("/api/v1/users/", "Users"),
        ("/api/v1/patients/", "Patients"),
        ("/api/v1/ecgs/", "ECGs"),
        ("/api/v1/analyses/", "Analyses"),
    ]
    
    # Testar cada endpoint
    results = []
    for path, description in endpoints:
        url = f"{base_url}{path}"
        status, data = test_endpoint(url)
        results.append((path, description, status, data))
        
        # Exibir resultado
        if status == 0:
            print(f"❌ {description:30} - {data['error']}")
        elif status == 200:
            print(f"✅ {description:30} - OK (200)")
        elif status == 404:
            print(f"⚠️  {description:30} - Not Found (404)")
        elif status == 401:
            print(f"🔒 {description:30} - Unauthorized (401)")
        elif status == 405:
            print(f"🚫 {description:30} - Method Not Allowed (405)")
        else:
            print(f"❓ {description:30} - Status {status}")
    
    # Verificar se o servidor está rodando
    print("\n" + "=" * 60)
    server_running = any(r[2] != 0 for r in results)
    
    if not server_running:
        print("❌ SERVIDOR NÃO ESTÁ RODANDO!")
        print("\n📝 Para iniciar o servidor:")
        print("1. Abra um terminal")
        print("2. Navegue até: backend/")
        print("3. Execute: uvicorn app.main:app --reload")
        return
    
    # Encontrar endpoints funcionais
    working_endpoints = [r for r in results if r[2] == 200]
    
    if working_endpoints:
        print(f"\n✅ Endpoints funcionando ({len(working_endpoints)}):")
        for path, desc, _, _ in working_endpoints:
            print(f"   - {path:30} ({desc})")
    
    # Verificar documentação
    docs_endpoints = [r for r in results if "Swagger" in r[1] or "ReDoc" in r[1] or "OpenAPI" in r[1]]
    working_docs = [r for r in docs_endpoints if r[2] == 200]
    
    if working_docs:
        print(f"\n📚 Documentação disponível em:")
        for path, desc, _, _ in working_docs:
            print(f"   - http://localhost:8000{path}")
    else:
        print("\n⚠️  Nenhum endpoint de documentação está funcionando!")
        print("Isso pode indicar que:")
        print("1. A documentação está desabilitada")
        print("2. Os endpoints estão em URLs diferentes")
        print("3. Há um problema na configuração do FastAPI")
    
    # Análise detalhada da resposta root
    root_response = next((r for r in results if r[0] == "/"), None)
    if root_response and root_response[2] == 200:
        print(f"\n📋 Resposta do endpoint root:")
        print(json.dumps(root_response[3], indent=2))
    
    # Sugestões
    print("\n💡 Sugestões:")
    if not any(r[2] == 200 for r in results if "docs" in r[0].lower()):
        print("1. Verifique se o FastAPI está configurado com docs_url e redoc_url")
        print("2. Verifique o arquivo app/main.py para as configurações de documentação")
        print("3. Tente acessar diretamente via navegador")
    
    # Verificar configuração do FastAPI
    print("\n🔧 Para verificar a configuração do FastAPI:")
    print("python -c \"from app.main import app; print(f'docs_url: {app.docs_url}'); print(f'redoc_url: {app.redoc_url}'); print(f'openapi_url: {app.openapi_url}')\"")

if __name__ == "__main__":
    main()
