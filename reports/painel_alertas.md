# Painel de alertas — equipe médica

Total: **47** alertas · suprimidos por duplicidade: 11
· por severidade: {'ALTO': 18, 'MODERADO': 18, 'CRITICO': 6, 'INFO': 5}

| Sev. | Paciente | Momento | Fonte | Alerta | Destino | Conduta sugerida |
|---|---|---|---|---|---|---|
| CRITICO | P002 | 2026-11-02 00:00:00 | prescricao | Dose acima do máximo: dipirona | Médico plantonista + Enfermagem (imediato) | Suspender administração e confirmar com o prescritor/farmácia clínica. |
| CRITICO | P003 | 2026-11-02 03:00:00 | movimentacao | Possível queda detectada | Médico plantonista + Enfermagem (imediato) | Avaliar o paciente imediatamente; protocolo pós-queda (TC de crânio se anticoagulado). |
| CRITICO | P003 | 2026-11-02 03:00:00 | fusao | Risco multimodal CRITICO (escore 0.83) | Médico plantonista + Enfermagem (imediato) | Avaliar o paciente imediatamente; protocolo pós-queda (TC de crânio se anticoagulado). Reposicionar o paciente (mudança de decúbito) e avaliar pele (escala de Braden). Suspender administração e confirmar com o prescritor/farmácia clínica. |
| CRITICO | P002 | 2026-11-02 08:00:00 | fusao | Risco multimodal CRITICO (escore 0.84) | Médico plantonista + Enfermagem (imediato) | Suspender administração e confirmar com o prescritor/farmácia clínica. Confirmar com o médico assistente se a suspensão é intencional. |
| CRITICO | P003 | 2026-11-02 10:00:25 | video | Fisioterapia: queda | Médico plantonista + Enfermagem (imediato) | Interromper a sessão, avaliar o paciente e registrar evento adverso. |
| CRITICO | P002 | 2026-11-02 11:40:00 | sinais_vitais | NEWS2 ≥ 7 — resposta de emergência | Médico plantonista + Enfermagem (imediato) | Acionar time de resposta rápida; avaliação médica imediata; considerar transferência/escala de cuidado. |
| ALTO | P003 | 2026-11-01 11:00:00 | movimentacao | Imobilidade prolongada | Enfermagem do setor (até 15 min) | Reposicionar o paciente (mudança de decúbito) e avaliar pele (escala de Braden). |
| ALTO | P003 | 2026-11-01 12:00:00 | prescricao | Dose acima do máximo: enoxaparina | Enfermagem do setor (até 15 min) | Suspender administração e confirmar com o prescritor/farmácia clínica. |
| ALTO | P003 | 2026-11-01 12:00:00 | prescricao | Aumento abrupto de dose: enoxaparina | Enfermagem do setor (até 15 min) | Verificar justificativa clínica (ex.: dose terapêutica vs profilática). |
| ALTO | P003 | 2026-11-01 12:00:00 | prescricao | Duplicidade terapêutica (BRA) | Enfermagem do setor (até 15 min) | Revisar prescrição: manter apenas um agente da classe. |
| ALTO | P003 | 2026-11-01 12:00:00 | fusao | Risco multimodal ALTO (escore 0.52) | Enfermagem do setor (até 15 min) | Reposicionar o paciente (mudança de decúbito) e avaliar pele (escala de Braden). Suspender administração e confirmar com o prescritor/farmácia clínica. |
| ALTO | P003 | 2026-11-01 14:10:00 | sinais_vitais | Crise hipertensiva (PAS ≥ 180 mmHg) | Enfermagem do setor (até 15 min) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| ALTO | P002 | 2026-11-02 00:00:00 | prescricao | Suspensão abrupta de antibiotico: ceftriaxona | Enfermagem do setor (até 15 min) | Confirmar com o médico assistente se a suspensão é intencional. |
| ALTO | P002 | 2026-11-02 00:00:00 | fusao | Risco multimodal ALTO (escore 0.73) | Enfermagem do setor (até 15 min) | Suspender administração e confirmar com o prescritor/farmácia clínica. |
| ALTO | P003 | 2026-11-02 06:00:00 | prescricao | Dose acima do máximo: enoxaparina | Enfermagem do setor (até 15 min) | Suspender administração e confirmar com o prescritor/farmácia clínica. |
| ALTO | P003 | 2026-11-02 06:00:00 | prescricao | Duplicidade terapêutica (BRA) | Enfermagem do setor (até 15 min) | Revisar prescrição: manter apenas um agente da classe. |
| ALTO | P003 | 2026-11-02 06:00:00 | prescricao | Interação medicamentosa relevante | Enfermagem do setor (até 15 min) | Avaliar troca do AINE por analgésico sem efeito antiplaquetário. |
| ALTO | P002 | 2026-11-02 09:00:00 | audio | Termos críticos na consulta | Enfermagem do setor (até 15 min) | Médico assistente deve revisar a transcrição e investigar os sintomas relatados. |
| ALTO | P003 | 2026-11-02 09:00:00 | audio | Alteração vocal compatível com fadiga | Enfermagem do setor (até 15 min) | Avaliar fadiga (anemia, distúrbio hidroeletrolítico, sono) e ajustar intensidade da reabilitação. |
| ALTO | P003 | 2026-11-02 09:00:00 | audio | Alteração vocal compatível com disartria | Enfermagem do setor (até 15 min) | Avaliação fonoaudiológica e neurológica; disartria nova/pior pode indicar evento neurológico. |
| ALTO | P003 | 2026-11-02 10:00:18 | video | Fisioterapia: sem supervisao | Enfermagem do setor (até 15 min) | Garantir supervisão contínua durante exercícios com risco de queda. |
| ALTO | P002 | 2026-11-02 10:05:00 | sinais_vitais | NEWS2 ≥ 5 — resposta urgente | Enfermagem do setor (até 15 min) | Avaliação médica urgente; aumentar frequência de monitorização para no mínimo de hora em hora. |
| ALTO | P003 | 2026-11-02 14:40:00 | sinais_vitais | Crise hipertensiva (PAS ≥ 180 mmHg) | Enfermagem do setor (até 15 min) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| ALTO | P003 | 2026-11-02 14:40:00 | sinais_vitais | PA diastólica (mmHg) acima da faixa | Enfermagem do setor (até 15 min) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| MODERADO | P003 | 2026-11-01 14:10:00 | sinais_vitais | Frequência cardíaca (bpm): desvio abrupto do basal do paciente | Painel de monitoramento (próxima ronda) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| MODERADO | P003 | 2026-11-01 14:10:00 | sinais_vitais | PA diastólica (mmHg) acima da faixa | Painel de monitoramento (próxima ronda) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| MODERADO | P003 | 2026-11-01 14:25:00 | sinais_vitais | Padrão multivariado atípico (Isolation Forest) | Painel de monitoramento (próxima ronda) | Revisar tendência dos sinais vitais. |
| MODERADO | P002 | 2026-11-01 16:35:00 | sinais_vitais | Padrão multivariado atípico (Isolation Forest) | Painel de monitoramento (próxima ronda) | Revisar tendência dos sinais vitais. |
| MODERADO | P002 | 2026-11-02 00:00:00 | prescricao | Aumento abrupto de dose: dipirona | Painel de monitoramento (próxima ronda) | Verificar justificativa clínica (ex.: dose terapêutica vs profilática). |
| MODERADO | P003 | 2026-11-02 02:00:00 | movimentacao | Agitação noturna | Painel de monitoramento (próxima ronda) | Avaliar delirium (CAM-ICU), dor e risco de queda; grades elevadas. |
| MODERADO | P002 | 2026-11-02 08:00:00 | sinais_vitais | Padrão multivariado atípico (Isolation Forest) | Painel de monitoramento (próxima ronda) | Revisar tendência dos sinais vitais. |
| MODERADO | P002 | 2026-11-02 08:00:00 | sinais_vitais | Tendência de deterioração clínica | Painel de monitoramento (próxima ronda) | Alerta precoce: reavaliar paciente, considerar protocolo de sepse se houver foco infeccioso. |
| MODERADO | P003 | 2026-11-02 09:00:00 | audio | Termos críticos na consulta | Painel de monitoramento (próxima ronda) | Médico assistente deve revisar a transcrição e investigar os sintomas relatados. |
| MODERADO | P002 | 2026-11-02 09:25:00 | sinais_vitais | SpO₂ (%) abaixo da faixa | Painel de monitoramento (próxima ronda) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| MODERADO | P003 | 2026-11-02 10:00:02 | video | Fisioterapia: deficit amplitude | Painel de monitoramento (próxima ronda) | Reavaliar amplitude de movimento e dor. |
| MODERADO | P003 | 2026-11-02 10:00:02 | video | Fisioterapia: assimetria | Painel de monitoramento (próxima ronda) | Ajustar plano de reabilitação (fortalecimento do lado deficitário). |
| MODERADO | P003 | 2026-11-02 10:00:14 | video | Fisioterapia: tremor | Painel de monitoramento (próxima ronda) | Comunicar neurologia; avaliar fadiga muscular e medicações. |
| MODERADO | P002 | 2026-11-02 10:05:00 | sinais_vitais | Temperatura (°C) acima da faixa | Painel de monitoramento (próxima ronda) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| MODERADO | P002 | 2026-11-02 11:40:00 | sinais_vitais | Frequência respiratória (irpm) acima da faixa | Painel de monitoramento (próxima ronda) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| MODERADO | P002 | 2026-11-02 12:55:00 | sinais_vitais | Frequência cardíaca (bpm) acima da faixa | Painel de monitoramento (próxima ronda) | Confirmar aferição à beira-leito e reavaliar o paciente. |
| MODERADO | P003 | 2026-11-02 14:50:00 | sinais_vitais | Padrão multivariado atípico (Isolation Forest) | Painel de monitoramento (próxima ronda) | Revisar tendência dos sinais vitais. |
| MODERADO | P002 | 2026-11-02 16:45:00 | sinais_vitais | PA sistólica (mmHg) abaixo da faixa | Painel de monitoramento (próxima ronda) | Confirmar aferição à beira-leito e reavaliar o paciente. |

