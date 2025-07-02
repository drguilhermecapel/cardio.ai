describe('Teste de Verificação', () => {
  it('deve executar um teste simples', () => {
    expect(1 + 1).toBe(2)
  })

  it('deve verificar strings', () => {
    expect('CardioAI Pro').toContain('CardioAI')
  })

  it('deve verificar objetos', () => {
    const config = {
      app: 'CardioAI Pro',
      version: '1.0.0',
      medical: true
    }
    
    expect(config).toHaveProperty('medical', true)
  })
})
