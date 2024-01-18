.data
tab:	.space	1024
	.align	2
N:	.word	256

.text

.text
main:
	addiu $29, $29, -16 # nv = 2 + na = 2
	xor $8, $8, $8 # i = 0
	lui $9, 0x1001
	ori $9, $9, 1024 # @N
	lw $9, 0($9)
	lui $10, 0x1001 # @tab[0]
	addiu $12, $0, -1
	while_main:
	slt $11, $8, $9
	beq $11, $0, suite_main
	sll $11, $8, 2 # i * 4
	addu $11, $10, $11 # @tab[i]
	sw $12, 0($11) # tab[i] = -1
	addiu $8, $8, 1 # i++
	j while_main
suite_main:
endwhile:
# scanf("%d", &n)
	ori $2, $0, 5
	syscall
	andi $2, $2, 0xFF # valeur de n optilisée dans $2 ici
	ori $4, $2, 0 # n
	lui $5, 0x1001 # tab (rechargement car syscall avant)
	jal fib
	ori $4, $2, 0
	ori $2, $0, 1
	syscall
	addiu $29, $29, 16
	ori $2, $0, 10
	syscall

fib:
	addiu $29, $29, -20 # na = 2 + nv = 0 + nr = 2 + 31
	sw $31, 16($29)
	sw $16, 12($29)
	sw $17, 8($29)
	sw $4, 20($29)
	sw $5, 24($29)
	sll $16, $4, 2 # n * 4
	addu $16, $16, $5 # @tab[n] dans reg persistant
	lw $8, 0($16) # $8 <- tab[n]
	addiu $9, $0, -1
	beq $9, $8, cas_non_deja_calcule
	ori $2, $8, 0
	j epilogue_fib
cas_non_deja_calcule:
	bne $4, $0, cas_n_non_0
	ori $2, $0, 1
	sw $2, 0($16)
	j epilogue_fib
cas_n_non_0:
	ori $2, $0, 1
	bne $4, $2, cas_n_sup_1
	sw $2, 0($16)
	j epilogue_fib
cas_n_sup_1:
	# fib(n-1, tab)
	addiu $4, $4, -1 # n - 1
	jal fib	
	ori $17, $2, 0 # sauvegarde du resultat de fib(n-1,tab)
	# fib(n-2, tab)
	lw $4, 20($29) # lecture n dans la pile
	addiu $4, $4, -2 # n - 2
	lw $5, 24($29) # lecture tab dans la pile
	jal fib
	# tab[n] = fib(n-1, tab) + fib(n-2, tab)
	addu $2, $17, $2
	sw $2, 0($16)
epilogue_fib:
	lw $31, 16($29)
	lw $16, 12($29)
	lw $17, 8($29)
	addiu $29, $29, 20
	jr $31
