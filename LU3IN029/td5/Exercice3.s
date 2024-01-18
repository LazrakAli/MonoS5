.data
p:	.word	10
q:	.word	15

.text

	lui	$8, 0x1001
	lw	$9, 0($8)
	lw	$10, 4($8)
	subu	$11, $9, $10
#p<q
	bgez	$11, else
#calcule la somme des entiers compris entre p et q inclus
	ori	$12,$9,0
for:
	addi	$9, $9,1
	add	$12,$12,$9
	bne	$9,$10, for
#affichage 
	ori	$4,$12,0
	ori	$2,$0,1
	syscall
#exit()
	ori	$2,$0,10
	syscall

else:		
	bne	$9, $10, else2
	ori 	$4, $9, 0
	ori	$2, $0, 1
	syscall
#exit()
	ori	$2,$0,10
	syscall
else2:
#affichage p car p=q
	ori 	$4, $0, 0
	ori	$2, $0, 1
	syscall
#exit()
	ori	$2,$0,10
	syscall
	