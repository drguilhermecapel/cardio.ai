// Script para verificar se o ambiente está configurado corretamente
const fs = require('fs');
const path = require('path');

console.log('🔍 Verificando configuração do frontend...\n');

// Verificar arquivos necessários
const requiredFiles = [
  'package.json',
  'vitest.config.ts',
  'src/setupTests.ts',
  'tsconfig.json',
  'vite.config.ts'
];

let allOk = true;

requiredFiles.forEach(file => {
  if (fs.existsSync(file)) {
    console.log(`✅ ${file} encontrado`);
  } else {
    console.log(`❌ ${file} NÃO encontrado`);
    allOk = false;
  }
});

// Verificar package.json
console.log('\n📦 Verificando package.json...');
try {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  
  // Verificar scripts necessários
  const requiredScripts = ['test', 'test:coverage'];
  requiredScripts.forEach(script => {
    if (packageJson.scripts && packageJson.scripts[script]) {
      console.log(`✅ Script '${script}' encontrado`);
    } else {
      console.log(`❌ Script '${script}' NÃO encontrado`);
      allOk = false;
    }
  });
  
  // Verificar dependências de teste
  const testDeps = ['vitest', '@vitest/coverage-v8', '@testing-library/react'];
  testDeps.forEach(dep => {
    if ((packageJson.devDependencies && packageJson.devDependencies[dep]) || 
        (packageJson.dependencies && packageJson.dependencies[dep])) {
      console.log(`✅ Dependência '${dep}' encontrada`);
    } else {
      console.log(`❌ Dependência '${dep}' NÃO encontrada`);
      allOk = false;
    }
  });
  
} catch (error) {
  console.log('❌ Erro ao ler package.json:', error.message);
  allOk = false;
}

// Resultado final
console.log('\n' + '='.repeat(50));
if (allOk) {
  console.log('✅ Tudo configurado corretamente!');
  console.log('\nPróximos passos:');
  console.log('1. npm install (se ainda não fez)');
  console.log('2. npm run test:coverage');
} else {
  console.log('❌ Há problemas na configuração!');
  console.log('\nCorreções necessárias:');
  console.log('1. Verifique os arquivos faltantes');
  console.log('2. Atualize o package.json');
  console.log('3. Execute npm install');
}
